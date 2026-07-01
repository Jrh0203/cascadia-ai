#!/usr/bin/env python3
"""Certify byte-exact V3 interruption/resume against an uninterrupted twin."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import uuid
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_safetensors_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise ValueError(f"{path} is not a safetensors file")
        header_size = struct.unpack("<Q", header_size_raw)[0]
        header = json.loads(handle.read(header_size).decode())
        data = handle.read()
    digest = hashlib.sha256()
    digest.update(b"cascadia-canonical-safetensors-v1")
    for name in sorted(key for key in header if key != "__metadata__"):
        metadata = header[name]
        start, end = metadata["data_offsets"]
        if not 0 <= start <= end <= len(data):
            raise ValueError(f"{path} tensor {name} has invalid offsets")
        descriptor = {
            "name": name,
            "dtype": metadata["dtype"],
            "shape": metadata["shape"],
            "bytes": end - start,
        }
        digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
        digest.update(data[start:end])
    return digest.hexdigest()


def _latest(run: Path) -> Path:
    name = _read(run / "latest.json").get("checkpoint")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{run} has no latest checkpoint")
    checkpoint = run / "checkpoints" / name
    if not checkpoint.is_dir():
        raise ValueError(f"latest checkpoint is absent: {checkpoint}")
    return checkpoint


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def certify(interrupted: Path, resumed: Path, control: Path, output: Path) -> dict[str, object]:
    interrupted_report = _read(interrupted)
    resumed_report = _read(resumed / "engineering-training-report.json")
    control_report = _read(control / "engineering-training-report.json")
    resumed_manifest = _read(resumed / "run-manifest.json")
    control_manifest = _read(control / "run-manifest.json")
    resumed_checkpoint = _latest(resumed)
    control_checkpoint = _latest(control)
    interrupted_state = interrupted_report.get("state", {})
    resumed_state = resumed_report.get("state", {})
    control_state = control_report.get("state", {})
    file_hashes: dict[str, dict[str, object]] = {}
    state_identical = True
    for name in ("model.safetensors", "optimizer.safetensors"):
        resumed_path = resumed_checkpoint / name
        control_path = control_checkpoint / name
        pair = {
            "resumed_container_sha256": _sha256(resumed_path),
            "uninterrupted_container_sha256": _sha256(control_path),
            "resumed_tensor_state_sha256": _canonical_safetensors_sha256(resumed_path),
            "uninterrupted_tensor_state_sha256": _canonical_safetensors_sha256(control_path),
        }
        pair["container_byte_identical"] = (
            pair["resumed_container_sha256"] == pair["uninterrupted_container_sha256"]
        )
        pair["tensor_state_identical"] = (
            pair["resumed_tensor_state_sha256"]
            == pair["uninterrupted_tensor_state_sha256"]
        )
        state_identical &= bool(pair["tensor_state_identical"])
        file_hashes[name] = pair
    run_manifest_identical = resumed_manifest == control_manifest
    training_config = resumed_report.get("training_config", {})
    expected_steps = math.ceil(
        int(training_config.get("examples", 0))
        / int(training_config.get("logical_batch_size", 1))
    ) * int(training_config.get("epochs", 0))
    interrupted_step = interrupted_state.get("global_step")
    cursor_exact = (
        isinstance(interrupted_step, int)
        and 0 < interrupted_step < expected_steps
        and interrupted_report.get("metrics", {}).get("interrupted") is True
        and resumed_report.get("metrics", {}).get("skipped_resume_batches")
        == float(interrupted_step)
        and resumed_state.get("global_step")
        == control_state.get("global_step")
        == expected_steps
        and resumed_state.get("epoch") == control_state.get("epoch") == 1
    )
    passed = state_identical and run_manifest_identical and cursor_exact
    receipt = {
        "schema_id": "cascadia-v3-checkpoint-recovery-receipt-v1",
        "passed": passed,
        "checkpoint_exact_continuation": passed,
        "run_manifest_identical": run_manifest_identical,
        "loader_cursor_exact": cursor_exact,
        "model_and_optimizer_tensor_state_identical": state_identical,
        "interrupted_global_step": interrupted_step,
        "final_global_step": resumed_state.get("global_step"),
        "run_manifest_blake3": resumed_report.get("run_manifest_blake3"),
        "file_hashes": file_hashes,
        "resumed_checkpoint": str(resumed_checkpoint),
        "uninterrupted_checkpoint": str(control_checkpoint),
    }
    _write_atomic(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interrupted-report", type=Path, required=True)
    parser.add_argument("--resumed-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = certify(
        args.interrupted_report,
        args.resumed_run,
        args.control_run,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
