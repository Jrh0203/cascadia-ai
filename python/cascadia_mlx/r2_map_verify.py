"""Standalone schema-v2 R2-MAP checkpoint verifier.

This module intentionally does not import the trainer.  A verification adapter
supplies the fixed public prediction batch, allowing the final replay format to
land independently without coupling verification to John2's disk layout.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

import blake3
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from cascadia_mlx.checkpoint import (
    CheckpointError,
    R2MapCheckpointBundle,
    load_r2_map_checkpoint_bundle,
    set_r2_map_checkpoint_pointer,
    verify_loss_stream_prefix,
    verify_r2_map_checkpoint_files,
)
from cascadia_mlx.r2_map_contracts import (
    CAMPAIGN_ROOT,
    canonical_campaign_path,
    require_local_storage_authority,
)
from cascadia_mlx.r2_map_model import R2MapBatch, R2MapModel, R2MapModelConfig
from cascadia_mlx.r2_map_training_contract import R2MapAdapterStep

VERIFICATION_SCHEMA = "r2-map-checkpoint-verification-v2"


class R2MapVerificationAdapter(Protocol):
    """Format-independent source for the fixed panel and exact resume batch."""

    dataset_contract: dict[str, Any]

    def fixed_prediction_batch(self, panel_id: str) -> R2MapBatch: ...

    def training_batch(
        self, cursor: dict[str, Any], sampler_state: dict[str, Any]
    ) -> R2MapAdapterStep: ...


def _verify_next_batch_identity(
    adapter: R2MapVerificationAdapter,
    *,
    cursor: dict[str, Any],
    sampler_state: dict[str, Any],
    expected: str,
) -> str:
    actual = adapter.training_batch(cursor, sampler_state).batch.batch_identity
    if actual != expected:
        raise CheckpointError("R2-MAP exact next batch differs from resume state")
    return actual


def _verify_dataset_contract(
    adapter: R2MapVerificationAdapter, expected: Mapping[str, Any]
) -> None:
    if adapter.dataset_contract != dict(expected):
        raise CheckpointError("R2-MAP verification dataset contract differs")


def prediction_panel(model: R2MapModel, batch: R2MapBatch) -> dict[str, mx.array]:
    prediction = model(batch)
    tensors = {
        "action_scores": prediction.action_scores,
        "predicted_score_to_go": prediction.predicted_score_to_go,
        "predicted_score_components_to_go": prediction.predicted_score_components_to_go,
        "bootstrap_policy_logits": prediction.bootstrap_policy_logits,
        "opponent_tile_slot_logits": prediction.opponent_next_action.tile_slot_logits,
        "opponent_wildlife_slot_logits": prediction.opponent_next_action.wildlife_slot_logits,
        "opponent_draft_kind_logits": prediction.opponent_next_action.draft_kind_logits,
        "opponent_drafted_wildlife_logits": (
            prediction.opponent_next_action.drafted_wildlife_logits
        ),
        "opponent_replace_three_logits": prediction.opponent_next_action.replace_three_logits,
        "opponent_paid_wipe_count_logits": (
            prediction.opponent_next_action.paid_wipe_count_logits
        ),
        "opponent_paid_wipe_mask_logits": (
            prediction.opponent_next_action.paid_wipe_mask_logits
        ),
        "market_disposition_logits": prediction.market_survival.disposition_logits,
        "market_pair_survival_logits": prediction.market_survival.pair_survival_logits,
        "market_final_slot_logits": prediction.market_survival.final_slot_logits,
    }
    mx.eval(tensors)
    return tensors


def verify_r2_map_checkpoint(
    checkpoint_path: str | Path,
    *,
    run_dir: str | Path,
    adapter: R2MapVerificationAdapter,
    expected_identity: Mapping[str, Any] | None = None,
    mark_last_verified: bool = False,
) -> dict[str, Any]:
    """Recompute the fixed panel exactly and optionally advance last_verified."""
    checkpoint_path = Path(checkpoint_path)
    run_dir = Path(run_dir)
    manifest, state, expected_panel = verify_r2_map_checkpoint_files(
        checkpoint_path,
        expected_identity=expected_identity,
    )
    loss_path = run_dir / state.loss_stream["relative_path"]
    verify_loss_stream_prefix(loss_path, state.loss_stream)
    model = R2MapModel(R2MapModelConfig.from_dict(manifest["model_config"]))
    model.load_weights(str(checkpoint_path / "model.safetensors"))
    mx.eval(model.parameters())
    panel_id = manifest["prediction_panel"]["panel_id"]
    _verify_dataset_contract(adapter, state.dataset_contract)
    actual_panel = prediction_panel(model, adapter.fixed_prediction_batch(panel_id))
    next_batch_identity = _verify_next_batch_identity(
        adapter,
        cursor=state.cursor,
        sampler_state=state.sampler_state,
        expected=state.next_batch_identity,
    )
    if set(actual_panel) != set(expected_panel):
        raise CheckpointError("R2-MAP recomputed prediction panel names differ")
    tensor_digests: dict[str, str] = {}
    for name, expected in expected_panel.items():
        actual = actual_panel[name]
        mx.eval(actual, expected)
        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        if actual_array.dtype != expected_array.dtype or not np.array_equal(
            actual_array, expected_array, equal_nan=False
        ):
            raise CheckpointError(f"R2-MAP fixed prediction panel differs exactly: {name}")
        tensor_digests[name] = blake3.blake3(actual_array.tobytes(order="C")).hexdigest()
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "schema_id": VERIFICATION_SCHEMA,
        "checkpoint_id": checkpoint_path.name,
        "checkpoint_manifest_blake3": _file_blake3(checkpoint_path / "checkpoint.json"),
        "prediction_panel_id": panel_id,
        "dataset_contract_blake3": _canonical_blake3(state.dataset_contract),
        "prediction_tensor_blake3": tensor_digests,
        "loss_stream_offset_bytes": state.loss_stream["offset_bytes"],
        "loss_stream_prefix_blake3": state.loss_stream["prefix_blake3"],
        "exact_prediction_match": True,
        "next_batch_identity": next_batch_identity,
        "exact_next_batch_match": True,
    }
    receipt["verification_id"] = _canonical_blake3(receipt)
    receipt_path = run_dir / "verifications" / f"{checkpoint_path.name}.json"
    _write_json_atomic_fsync(receipt_path, receipt)
    if mark_last_verified:
        set_r2_map_checkpoint_pointer(
            run_dir,
            "last_verified",
            checkpoint_path,
            metadata={"verification_id": receipt["verification_id"]},
        )
    return receipt


def verify_r2_map_checkpoint_bundle_in_memory(
    bundle: R2MapCheckpointBundle,
    *,
    loss_content: bytes,
    adapter: R2MapVerificationAdapter,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the fixed panel without creating a local checkpoint or receipt."""
    loaded = load_r2_map_checkpoint_bundle(
        bundle,
        model_factory=lambda values: R2MapModel(R2MapModelConfig.from_dict(values)),
        optimizer_factory=lambda: optim.AdamW(learning_rate=1.0),
        expected_identity=expected_identity,
        loss_stream=loss_content,
    )
    panel_id = loaded.manifest["prediction_panel"]["panel_id"]
    _verify_dataset_contract(adapter, loaded.state.dataset_contract)
    actual_panel = prediction_panel(loaded.model, adapter.fixed_prediction_batch(panel_id))
    state = loaded.state
    next_batch_identity = _verify_next_batch_identity(
        adapter,
        cursor=state.cursor,
        sampler_state=state.sampler_state,
        expected=state.next_batch_identity,
    )
    if set(actual_panel) != set(loaded.prediction_panel):
        raise CheckpointError("R2-MAP recomputed prediction panel names differ")
    tensor_digests: dict[str, str] = {}
    for name, expected in loaded.prediction_panel.items():
        actual = actual_panel[name]
        mx.eval(actual, expected)
        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        if actual_array.dtype != expected_array.dtype or not np.array_equal(
            actual_array, expected_array, equal_nan=False
        ):
            raise CheckpointError(f"R2-MAP fixed prediction panel differs exactly: {name}")
        tensor_digests[name] = blake3.blake3(actual_array.tobytes(order="C")).hexdigest()
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "schema_id": VERIFICATION_SCHEMA,
        "checkpoint_id": bundle.checkpoint_id,
        "checkpoint_manifest_blake3": bundle.manifest_blake3,
        "prediction_panel_id": panel_id,
        "dataset_contract_blake3": _canonical_blake3(state.dataset_contract),
        "prediction_tensor_blake3": tensor_digests,
        "loss_stream_offset_bytes": state.loss_stream["offset_bytes"],
        "loss_stream_prefix_blake3": state.loss_stream["prefix_blake3"],
        "exact_prediction_match": True,
        "next_batch_identity": next_batch_identity,
        "exact_next_batch_match": True,
    }
    receipt["verification_id"] = _canonical_blake3(receipt)
    return receipt


def verify_integrity_only(
    checkpoint_path: str | Path,
    *,
    run_dir: str | Path,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check bytes and loss binding without claiming prediction verification."""
    manifest, state, _ = verify_r2_map_checkpoint_files(
        checkpoint_path,
        expected_identity=expected_identity,
    )
    verify_loss_stream_prefix(Path(run_dir) / state.loss_stream["relative_path"], state.loss_stream)
    return {
        "schema_version": 1,
        "checkpoint_id": manifest["checkpoint_id"],
        "integrity_verified": True,
        "fixed_prediction_recomputed": False,
        "last_verified_pointer_advanced": False,
    }


def validate_verification_receipt(
    path: str | Path,
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Reject forged, stale, or identity-drifted fixed-panel verification receipts."""
    path = Path(path)
    checkpoint_path = Path(checkpoint_path)
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read R2-MAP verification receipt: {error}") from error
    _, state, _ = verify_r2_map_checkpoint_files(checkpoint_path)
    return validate_verification_receipt_value(
        receipt,
        checkpoint_id=checkpoint_path.name,
        checkpoint_manifest_blake3=_file_blake3(checkpoint_path / "checkpoint.json"),
        expected_dataset_contract_blake3=_canonical_blake3(state.dataset_contract),
    )


def validate_verification_receipt_value(
    value: Mapping[str, Any],
    *,
    checkpoint_id: str,
    checkpoint_manifest_blake3: str,
    expected_dataset_contract_blake3: str | None = None,
) -> dict[str, Any]:
    """Validate an in-memory receipt before publishing or advancing a pointer."""
    receipt = dict(value)
    claimed = receipt.pop("verification_id", None)
    if (
        receipt.get("schema_version") != 2
        or receipt.get("schema_id") != VERIFICATION_SCHEMA
        or receipt.get("checkpoint_id") != checkpoint_id
        or receipt.get("checkpoint_manifest_blake3")
        != checkpoint_manifest_blake3
        or receipt.get("exact_prediction_match") is not True
        or receipt.get("exact_next_batch_match") is not True
        or not isinstance(receipt.get("next_batch_identity"), str)
        or not receipt["next_batch_identity"]
        or not isinstance(receipt.get("dataset_contract_blake3"), str)
        or len(receipt["dataset_contract_blake3"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt["dataset_contract_blake3"]
        )
        or (
            expected_dataset_contract_blake3 is not None
            and receipt["dataset_contract_blake3"] != expected_dataset_contract_blake3
        )
        or claimed != _canonical_blake3(receipt)
    ):
        raise CheckpointError("R2-MAP verification receipt identity differs")
    receipt["verification_id"] = claimed
    return receipt


def compare_r2_map_checkpoint_tensors(
    left_checkpoint: str | Path,
    right_checkpoint: str | Path,
) -> dict[str, Any]:
    """Prove semantic model/optimizer equality across different run identities."""
    left_checkpoint = Path(left_checkpoint)
    right_checkpoint = Path(right_checkpoint)
    verify_r2_map_checkpoint_files(left_checkpoint)
    verify_r2_map_checkpoint_files(right_checkpoint)
    bundles: dict[str, dict[str, Any]] = {}
    for file_name in ("model.safetensors", "optimizer.safetensors"):
        left = mx.load(str(left_checkpoint / file_name))
        right = mx.load(str(right_checkpoint / file_name))
        if set(left) != set(right):
            raise CheckpointError(f"R2-MAP {file_name} tensor names differ")
        digest = blake3.blake3()
        for name in sorted(left):
            left_array = np.asarray(left[name])
            right_array = np.asarray(right[name])
            if (
                left_array.dtype != right_array.dtype
                or left_array.shape != right_array.shape
                or not np.array_equal(left_array, right_array, equal_nan=False)
            ):
                raise CheckpointError(f"R2-MAP {file_name} tensor differs: {name}")
            name_bytes = name.encode()
            dtype_bytes = left_array.dtype.str.encode()
            digest.update(len(name_bytes).to_bytes(4, "little"))
            digest.update(name_bytes)
            digest.update(len(dtype_bytes).to_bytes(2, "little"))
            digest.update(dtype_bytes)
            digest.update(len(left_array.shape).to_bytes(2, "little"))
            for dimension in left_array.shape:
                digest.update(int(dimension).to_bytes(8, "little"))
            digest.update(left_array.tobytes(order="C"))
        bundles[file_name] = {
            "tensor_count": len(left),
            "semantic_blake3": digest.hexdigest(),
            "exact_match": True,
        }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": "r2-map-checkpoint-semantic-parity-v1",
        "left_checkpoint": left_checkpoint.name,
        "right_checkpoint": right_checkpoint.name,
        "bundles": bundles,
        "exact_match": True,
    }
    receipt["parity_id"] = _canonical_blake3(receipt)
    return receipt


def _canonical_blake3(value: Any) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _file_blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic_fsync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--train-stream", type=Path)
    parser.add_argument("--validation-stream", type=Path)
    parser.add_argument("--panel-stream", type=Path)
    parser.add_argument("--mark-last-verified", action="store_true")
    parser.add_argument("--compare-checkpoint", type=Path)
    arguments = parser.parse_args()
    require_local_storage_authority()
    canonical_campaign_path(
        arguments.run_dir,
        root=CAMPAIGN_ROOT,
        label="verification run directory",
    )
    canonical_campaign_path(
        arguments.checkpoint,
        root=CAMPAIGN_ROOT,
        label="verification checkpoint",
    )
    if arguments.compare_checkpoint is not None:
        canonical_campaign_path(
            arguments.compare_checkpoint,
            root=CAMPAIGN_ROOT,
            label="comparison checkpoint",
        )
    for label, path in (
        ("verification manifest", arguments.manifest),
        ("verification train stream", arguments.train_stream),
        ("verification validation stream", arguments.validation_stream),
        ("verification panel stream", arguments.panel_stream),
    ):
        if path is not None:
            canonical_campaign_path(path, root=CAMPAIGN_ROOT, label=label)
    try:
        dataset_arguments = (
            arguments.manifest,
            arguments.train_stream,
            arguments.validation_stream,
            arguments.panel_stream,
        )
        if arguments.compare_checkpoint is not None:
            if arguments.integrity_only or any(dataset_arguments):
                parser.error("checkpoint comparison forbids integrity and dataset arguments")
            result = compare_r2_map_checkpoint_tensors(
                arguments.checkpoint,
                arguments.compare_checkpoint,
            )
        elif arguments.integrity_only:
            if any(dataset_arguments) or arguments.mark_last_verified:
                parser.error("integrity-only forbids dataset and pointer arguments")
            result = verify_integrity_only(arguments.checkpoint, run_dir=arguments.run_dir)
        else:
            if not all(dataset_arguments):
                parser.error(
                    "full verification requires --manifest, --train-stream, "
                    "--validation-stream, and --panel-stream"
                )
            from cascadia_mlx.r2_map_dataset import R2MapDatasetAdapter

            with R2MapDatasetAdapter.open(
                train_manifest=arguments.manifest,
                train_stream=arguments.train_stream,
                validation_manifest=arguments.manifest,
                validation_stream=arguments.validation_stream,
                panel_manifest=arguments.manifest,
                panel_stream=arguments.panel_stream,
            ) as adapter:
                result = verify_r2_map_checkpoint(
                    arguments.checkpoint,
                    run_dir=arguments.run_dir,
                    adapter=adapter,
                    mark_last_verified=arguments.mark_last_verified,
                )
    except CheckpointError as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
