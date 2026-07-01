"""Verify one trained R2-MAP checkpoint across MLX and NumPy inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.checkpoint import verify_r2_map_checkpoint_files
from cascadia_mlx.r2_map_market_decision import decode_market_decision_action_bytes
from cascadia_mlx.r2_map_model import (
    R2MapMarketDecisionBatch,
    R2MapModel,
    R2MapModelConfig,
    R2MapPublicState,
)
from cascadia_mlx.r2_map_numpy import R2MapNumpyModel
from cascadia_mlx.r2_map_pipe_dataset import R2MapPackedPipeDatasetAdapter
from cascadia_mlx.r2_map_verify import prediction_panel, validate_verification_receipt

SCHEMA_ID = "cascadia.r2-map.mlx-numpy-checkpoint-parity.v1"
TOLERANCE = 2e-5
PUBLIC_FIELDS = (
    "token_features",
    "token_types",
    "token_mask",
    "market_features",
    "market_mask",
    "player_features",
    "player_mask",
    "global_features",
)


class BackendParityError(RuntimeError):
    pass


def _file_blake3(path: Path) -> str:
    hasher = blake3.blake3()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _receipt_identity(value: dict[str, Any]) -> str:
    hasher = blake3.blake3(b"r2-map-mlx-numpy-checkpoint-parity-receipt-v1")
    for name in (
        "schema_id",
        "checkpoint_id",
        "checkpoint_manifest_blake3",
        "model_weights_blake3",
        "verification_id",
    ):
        encoded = str(value[name]).encode("ascii")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
    hasher.update(bytes([bool(value["finite"]), bool(value["passed"])]))
    return hasher.hexdigest()


def _numpy_state(state: R2MapPublicState, index: int, *, candidate: bool) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in PUBLIC_FIELDS:
        value = np.asarray(getattr(state, name))
        result[name] = value[index] if candidate else value[index : index + 1]
    return result


def _maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise BackendParityError("backend prediction shape or finiteness differs")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)), initial=0.0))


def compare_checkpoint(
    *,
    checkpoint: Path,
    compact_index: Path,
    compact_shard_root: Path,
    compact_exporter: Path,
    validated_aggregate_receipt: Path,
    validated_packing_receipt: Path,
) -> dict[str, Any]:
    run_dir = checkpoint.parent.parent
    manifest, _state, _panel = verify_r2_map_checkpoint_files(checkpoint)
    verification_path = run_dir / "verifications" / f"{checkpoint.name}.json"
    verification = validate_verification_receipt(verification_path, checkpoint_path=checkpoint)
    model = R2MapModel(R2MapModelConfig.from_dict(manifest["model_config"]))
    model.load_weights(str(checkpoint / "model.safetensors"))
    model.eval()
    mx.eval(model.parameters())
    portable = R2MapNumpyModel(
        checkpoint / "model.safetensors",
        manifest["model_config"],
        candidate_chunk_size=256,
    )
    with R2MapPackedPipeDatasetAdapter(
        index=compact_index,
        shard_root=compact_shard_root,
        exporter=compact_exporter,
        validated_aggregate_receipt=validated_aggregate_receipt,
        validated_packing_receipt=validated_packing_receipt,
        group_batch_size=256,
        maximum_candidates_per_batch=256,
        sampler_seed=20260618,
    ) as adapter:
        batch = adapter.fixed_prediction_batch(manifest["prediction_panel"]["panel_id"])

    mlx_panel = prediction_panel(model, batch)
    mx.eval(mlx_panel)
    action_errors = {
        "action_scores": 0.0,
        "predicted_score_to_go": 0.0,
        "predicted_score_components_to_go": 0.0,
        "bootstrap_policy_logits": 0.0,
    }
    groups, _ = batch.validate()
    for group in range(groups):
        valid = np.asarray(batch.candidate_mask[group], dtype=np.bool_)
        portable_prediction = portable.score_action_features(
            _numpy_state(batch.parent, group, candidate=False),
            {
                name: value[valid]
                for name, value in _numpy_state(batch.candidates, group, candidate=True).items()
            },
            np.asarray(batch.action_features[group])[valid],
            np.asarray(batch.exact_afterstate_scores[group])[valid],
        )
        for name in action_errors:
            mlx_value = np.asarray(mlx_panel[name][group])[valid]
            numpy_value = np.asarray(getattr(portable_prediction, name))
            action_errors[name] = max(action_errors[name], _maximum_error(mlx_value, numpy_value))

    market_bytes = np.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 2, 0, 0, 0, 0, 0],
            [1, 1, 3, 5, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    first_parent = R2MapPublicState(
        **{name: getattr(batch.parent, name)[:1] for name in PUBLIC_FIELDS}
    )
    market_batch = R2MapMarketDecisionBatch(
        public_state=first_parent,
        action_mask=mx.ones((1, len(market_bytes)), dtype=mx.bool_),
        action_features=mx.array(decode_market_decision_action_bytes(market_bytes)[None, ...]),
        exact_current_scores=mx.array([50.0], dtype=mx.float32),
    )
    mlx_market = model.score_market_decisions(market_batch)
    mx.eval(mlx_market.action_scores, mlx_market.predicted_score_to_go)
    numpy_market = portable.score_market_decisions(
        _numpy_state(first_parent, 0, candidate=False), market_bytes, 50.0
    )
    market_errors = {
        "action_scores": _maximum_error(
            np.asarray(mlx_market.action_scores)[0], numpy_market.action_scores
        ),
        "predicted_score_to_go": _maximum_error(
            np.asarray(mlx_market.predicted_score_to_go)[0],
            numpy_market.predicted_score_to_go,
        ),
    }
    maximum_error = max(*action_errors.values(), *market_errors.values())
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "schema_id": SCHEMA_ID,
        "checkpoint_id": checkpoint.name,
        "checkpoint_manifest_blake3": _file_blake3(checkpoint / "checkpoint.json"),
        "model_weights_blake3": _file_blake3(checkpoint / "model.safetensors"),
        "verification_id": verification["verification_id"],
        "fixed_panel_groups": groups,
        "action_maximum_absolute_errors": action_errors,
        "market_maximum_absolute_errors": market_errors,
        "maximum_absolute_error": maximum_error,
        "tolerance": TOLERANCE,
        "finite": True,
        "passed": maximum_error <= TOLERANCE,
    }
    receipt["receipt_blake3"] = _receipt_identity(receipt)
    if not receipt["passed"]:
        raise BackendParityError(
            f"MLX/NumPy maximum absolute error {maximum_error} exceeds {TOLERANCE}"
        )
    return receipt


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise BackendParityError(f"refusing to replace different parity receipt: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compact-index", type=Path, required=True)
    parser.add_argument("--compact-shard-root", type=Path, required=True)
    parser.add_argument("--compact-exporter", type=Path, required=True)
    parser.add_argument("--validated-aggregate-receipt", type=Path, required=True)
    parser.add_argument("--validated-packing-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        receipt = compare_checkpoint(
            checkpoint=arguments.checkpoint.resolve(strict=True),
            compact_index=arguments.compact_index,
            compact_shard_root=arguments.compact_shard_root,
            compact_exporter=arguments.compact_exporter,
            validated_aggregate_receipt=arguments.validated_aggregate_receipt,
            validated_packing_receipt=arguments.validated_packing_receipt,
        )
        _write_atomic(arguments.output, receipt)
    except (BackendParityError, OSError, ValueError) as error:
        print(f"R2-MAP backend parity refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
