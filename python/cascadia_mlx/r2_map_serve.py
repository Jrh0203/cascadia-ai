"""Long-lived local grouped reference service for exhaustive R2-MAP scoring.

The reference protocol is intentionally explicit and copy-heavy.  It validates
every tensor, group offset, checkpoint identity, and ordered action identity
before scoring all supplied legal actions exactly once.  Shared memory,
candidate chunks, staged encodes, and other performance work belong to the
post-reference optimization program.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import blake3
import numpy as np

try:
    import mlx.core as mx

    from cascadia_mlx.checkpoint import verify_r2_map_checkpoint_files
    from cascadia_mlx.r2_map_model import (
        R2MapBatch,
        R2MapMarketDecisionBatch,
        R2MapModel,
        R2MapModelConfig,
        R2MapPublicState,
    )
    from cascadia_mlx.r2_map_verify import validate_verification_receipt

    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False

from cascadia_mlx.r2_map_market_decision import (
    MARKET_DECISION_ACTION_SIZE,
    MarketDecisionKind,
    market_decision_action_id,
    validate_canonical_market_action_order,
)
from cascadia_mlx.r2_map_numpy import (
    ACTION_BYTES as GRADED_ORACLE_ACTION_FEATURE_SIZE,
)
from cascadia_mlx.r2_map_numpy import (
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    DEFAULT_CANDIDATE_CHUNK_SIZE,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
    R2MapNumpyModel,
    decode_action_features,
    decode_market_action_features,
)

PROTOCOL_MAGIC = b"R2MP"
PROTOCOL_VERSION = 3
REQUEST_SCHEMA = "r2-map-grouped-exhaustive-request-v3"
RESPONSE_SCHEMA = "r2-map-grouped-exhaustive-response-v3"
MESSAGE_SCORE_GROUPS = 0x20
MESSAGE_SCORE_MARKET_DECISIONS = 0x21
MESSAGE_SHUTDOWN = 0x02
MESSAGE_SCORE_RESPONSE = 0x8020
MESSAGE_SCORE_MARKET_DECISIONS_RESPONSE = 0x8021
MESSAGE_ERROR = 0xFFFF
FRAME_HEADER = struct.Struct("<4sHHIII")
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_TENSOR_BYTES = 1024 * 1024 * 1024
MAX_GROUPS = 16
# Maximum observed complete-action width in the frozen full-legal evidence.
REFERENCE_MAX_CANDIDATES_PER_GROUP = 6_372
# A protocol safety bound, deliberately wider than the frozen reference panel.
# Crossing it rejects the whole request; no action set is ever truncated.
PROTOCOL_MAX_CANDIDATES_PER_GROUP = 8_192
MAX_TOTAL_CANDIDATES = MAX_GROUPS * PROTOCOL_MAX_CANDIDATES_PER_GROUP
DEFAULT_REGISTRY_CAPACITY = 4
SERVING_BUNDLE_SCHEMA = "r2-map-local-serving-bundle-v2"

REQUEST_TENSOR_DTYPES = {
    "candidate_offsets": "<i4",
    "parent_token_features": "<f4",
    "parent_token_types": "<i4",
    "parent_token_mask": "|u1",
    "parent_market_features": "<f4",
    "parent_market_mask": "|u1",
    "parent_player_features": "<f4",
    "parent_player_mask": "|u1",
    "parent_global_features": "<f4",
    "candidate_token_features": "<f4",
    "candidate_token_types": "<i4",
    "candidate_token_mask": "|u1",
    "candidate_market_features": "<f4",
    "candidate_market_mask": "|u1",
    "candidate_player_features": "<f4",
    "candidate_player_mask": "|u1",
    "candidate_global_features": "<f4",
    "action_bytes": "|u1",
    "exact_afterstate_scores": "<f4",
}

RESPONSE_TENSOR_DTYPES = {
    "action_scores": "<f4",
    "predicted_score_to_go": "<f4",
    "predicted_score_components_to_go": "<f4",
    "bootstrap_policy_logits": "<f4",
}

MARKET_REQUEST_SCHEMA = "r2-map-public-market-decision-request-v3"
MARKET_RESPONSE_SCHEMA = "r2-map-public-market-decision-response-v2"
MARKET_REQUEST_TENSOR_DTYPES = {
    "action_offsets": "<i4",
    "parent_token_features": "<f4",
    "parent_token_types": "<i4",
    "parent_token_mask": "|u1",
    "parent_market_features": "<f4",
    "parent_market_mask": "|u1",
    "parent_player_features": "<f4",
    "parent_player_mask": "|u1",
    "parent_global_features": "<f4",
    "action_bytes": "|u1",
    "exact_current_scores": "<f4",
}
MARKET_RESPONSE_TENSOR_DTYPES = {
    "market_action_scores": "<f4",
    "market_predicted_score_to_go": "<f4",
}

REQUEST_SCHEMA_BLAKE3 = blake3.blake3(
    json.dumps(
        {
            "schema": REQUEST_SCHEMA,
            "tensor_dtypes": REQUEST_TENSOR_DTYPES,
            "maximum_groups": MAX_GROUPS,
            "reference_maximum_candidates_per_group": REFERENCE_MAX_CANDIDATES_PER_GROUP,
            "protocol_maximum_candidates_per_group": PROTOCOL_MAX_CANDIDATES_PER_GROUP,
            "board_slots": BOARD_SLOTS,
            "board_token_capacity": BOARD_TOKEN_CAPACITY,
            "token_features": TOKEN_FEATURES,
            "exhaustive": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
MARKET_REQUEST_SCHEMA_BLAKE3 = blake3.blake3(
    json.dumps(
        {
            "schema": MARKET_REQUEST_SCHEMA,
            "tensor_dtypes": MARKET_REQUEST_TENSOR_DTYPES,
            "maximum_groups": MAX_GROUPS,
            "maximum_actions_per_group": 16,
            "board_slots": BOARD_SLOTS,
            "board_token_capacity": BOARD_TOKEN_CAPACITY,
            "token_features": TOKEN_FEATURES,
            "exhaustive": True,
            "public_pre_refill_only": True,
            "public_universal_legality_fields": [
                "public_wildlife_bag_counts",
                "public_market_wildlife",
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()
MARKET_RESPONSE_SCHEMA_BLAKE3 = blake3.blake3(
    json.dumps(
        {
            "schema": MARKET_RESPONSE_SCHEMA,
            "tensor_dtypes": MARKET_RESPONSE_TENSOR_DTYPES,
            "group_fields": [
                "group_id",
                "decision_id",
                "model",
                "action_offset",
                "action_count",
                "action_ids",
                "ordered_action_ids_blake3",
                "decision_kind",
                "public_nature_tokens",
                "public_wildlife_bag_total",
                "public_wildlife_bag_counts",
                "public_market_wildlife",
                "diagnostics",
            ],
            "group_diagnostics": [
                "actions_enumerated",
                "actions_scored",
                "complete_cardinality",
                "hidden_refill_inputs",
            ],
            "exhaustive": True,
            "public_pre_refill_only": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


class R2MapProtocolError(ValueError):
    """The local grouped frame is malformed or incomplete."""


@dataclass(frozen=True)
class R2MapRegistryEntry:
    checkpoint_id: str
    checkpoint_manifest_blake3: str
    model_config_blake3: str
    model_weights_blake3: str
    verification_id: str
    model: Any

    def identity(self) -> dict[str, str]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_manifest_blake3": self.checkpoint_manifest_blake3,
            "model_config_blake3": self.model_config_blake3,
            "model_weights_blake3": self.model_weights_blake3,
            "verification_id": self.verification_id,
        }


class R2MapCheckpointRegistry:
    """Bounded verified local model registry with cohort-boundary LRU eviction."""

    def __init__(
        self,
        capacity: int = DEFAULT_REGISTRY_CAPACITY,
        *,
        expected_protocols: Mapping[str, Any] | None = None,
        backend: str = "auto",
        candidate_chunk_size: int = DEFAULT_CANDIDATE_CHUNK_SIZE,
    ):
        if capacity <= 0:
            raise ValueError("R2-MAP registry capacity must be positive")
        if backend == "auto":
            backend = "mlx" if MLX_AVAILABLE else "numpy"
        if backend not in {"mlx", "numpy"}:
            raise ValueError("R2-MAP backend must be auto, mlx, or numpy")
        if backend == "mlx" and not MLX_AVAILABLE:
            raise ValueError("MLX backend requested where MLX is unavailable")
        if candidate_chunk_size <= 0:
            raise ValueError("R2-MAP candidate chunk size must be positive")
        self.capacity = capacity
        self.backend = backend
        self.candidate_chunk_size = candidate_chunk_size
        self._entries: OrderedDict[str, R2MapRegistryEntry] = OrderedDict()
        self._pinned: set[str] = set()
        self._active_wave = False
        self.protocols = (
            None if expected_protocols is None else _validate_protocol_identity(expected_protocols)
        )
        self.closed = False

    def register_model(self, entry: R2MapRegistryEntry, *, pinned: bool = False) -> None:
        self._require_open()
        _validate_model_identity(entry.identity())
        existing = self._entries.get(entry.checkpoint_id)
        if existing is not None:
            if existing.identity() != entry.identity():
                raise R2MapProtocolError("checkpoint id is already bound to different model bytes")
            if existing.model is not entry.model:
                raise R2MapProtocolError("checkpoint id cannot replace its resident model instance")
            self._entries.move_to_end(entry.checkpoint_id)
            if pinned:
                self._pinned.add(entry.checkpoint_id)
            return
        while len(self._entries) >= self.capacity:
            if self._active_wave:
                raise R2MapProtocolError("registry cannot evict during an inference wave")
            evicted = next(
                (key for key in self._entries if key not in self._pinned),
                None,
            )
            if evicted is None:
                raise R2MapProtocolError("registry capacity is exhausted by pinned checkpoints")
            del self._entries[evicted]
        self._entries[entry.checkpoint_id] = entry
        if pinned:
            self._pinned.add(entry.checkpoint_id)

    def register_verified_checkpoint(
        self,
        *,
        run_dir: str | Path,
        checkpoint_path: str | Path,
        pinned: bool = False,
    ) -> R2MapRegistryEntry:
        run_dir = Path(run_dir)
        checkpoint_path = Path(checkpoint_path)
        if self.backend == "mlx":
            manifest, _, _ = verify_r2_map_checkpoint_files(checkpoint_path)
            receipt = validate_verification_receipt(
                run_dir / "verifications" / f"{checkpoint_path.name}.json",
                checkpoint_path=checkpoint_path,
            )
            model = R2MapModel(R2MapModelConfig.from_dict(manifest["model_config"]))
            model.load_weights(str(checkpoint_path / "model.safetensors"))
            model.eval()
            mx.eval(model.parameters())
        else:
            manifest, receipt = _verify_portable_checkpoint(run_dir, checkpoint_path)
            model = R2MapNumpyModel(
                checkpoint_path / "model.safetensors",
                manifest["model_config"],
                candidate_chunk_size=self.candidate_chunk_size,
            )
        entry = R2MapRegistryEntry(
            checkpoint_id=checkpoint_path.name,
            checkpoint_manifest_blake3=_file_blake3(checkpoint_path / "checkpoint.json"),
            model_config_blake3=manifest["identity"]["model_config_blake3"],
            model_weights_blake3=manifest["files"]["model.safetensors"]["blake3"],
            verification_id=receipt["verification_id"],
            model=model,
        )
        self.register_model(entry, pinned=pinned)
        return entry

    def register_verified_bundle(self, path: str | Path) -> None:
        """Load a strict multi-run bundle and verify every bound identity."""
        bundle_path = Path(path)
        try:
            bundle = json.loads(bundle_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise R2MapProtocolError(f"cannot read local serving bundle: {error}") from error
        if (
            not isinstance(bundle, dict)
            or set(bundle) != {"schema_version", "schema_id", "protocols", "entries"}
            or bundle.get("schema_version") != 2
            or bundle.get("schema_id") != SERVING_BUNDLE_SCHEMA
            or not isinstance(bundle.get("entries"), list)
            or not bundle["entries"]
        ):
            raise R2MapProtocolError("local serving bundle schema differs")
        protocols = _validate_protocol_identity(bundle["protocols"])
        if self.protocols is not None and protocols != self.protocols:
            raise R2MapProtocolError("local serving bundle protocol identity is stale")
        compact_identities: set[str] = set()
        checkpoint_ids: set[str] = set()
        for raw in bundle["entries"]:
            required = {
                "manifest_identity_blake3",
                "run_dir",
                "checkpoint_path",
                "model",
                "pinned",
            }
            if not isinstance(raw, dict) or set(raw) != required:
                raise R2MapProtocolError("local serving bundle entry fields differ")
            compact = raw["manifest_identity_blake3"]
            model_identity = raw["model"]
            _validate_model_identity(model_identity)
            if (
                not _is_blake3(compact)
                or compact in compact_identities
                or model_identity["checkpoint_id"] in checkpoint_ids
            ):
                raise R2MapProtocolError("local serving bundle identity is invalid or duplicated")
            run_dir = Path(raw["run_dir"])
            checkpoint_path = Path(raw["checkpoint_path"])
            if not run_dir.is_absolute() or not checkpoint_path.is_absolute():
                raise R2MapProtocolError("local serving bundle paths must be absolute")
            if self.backend == "mlx":
                manifest, _, _ = verify_r2_map_checkpoint_files(checkpoint_path)
            else:
                manifest, _receipt = _verify_portable_checkpoint(run_dir, checkpoint_path)
            if manifest["manifest_identity_blake3"] != compact:
                raise R2MapProtocolError("compact policy identity differs from checkpoint manifest")
            entry = self.register_verified_checkpoint(
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                pinned=bool(raw["pinned"]),
            )
            if entry.identity() != model_identity:
                raise R2MapProtocolError("bundle model identity differs from verified checkpoint")
            compact_identities.add(compact)
            checkpoint_ids.add(entry.checkpoint_id)
        self.protocols = protocols

    def get(self, identity: Mapping[str, Any]) -> R2MapRegistryEntry:
        self._require_open()
        _validate_model_identity(identity)
        checkpoint_id = str(identity["checkpoint_id"])
        try:
            entry = self._entries[checkpoint_id]
        except KeyError as error:
            raise R2MapProtocolError(f"checkpoint is not resident: {checkpoint_id}") from error
        if entry.identity() != dict(identity):
            raise R2MapProtocolError("request checkpoint hashes differ from the resident model")
        self._entries.move_to_end(checkpoint_id)
        return entry

    def pin(self, checkpoint_id: str) -> None:
        self._require_open()
        if checkpoint_id not in self._entries:
            raise R2MapProtocolError("cannot pin a nonresident checkpoint")
        self._pinned.add(checkpoint_id)

    def unpin(self, checkpoint_id: str) -> None:
        self._require_open()
        self._pinned.discard(checkpoint_id)

    def begin_wave(self) -> None:
        self._require_open()
        if self._active_wave:
            raise R2MapProtocolError("nested registry inference wave")
        self._active_wave = True

    def end_wave(self) -> None:
        self._active_wave = False

    def close(self) -> None:
        self._entries.clear()
        self._pinned.clear()
        self._active_wave = False
        self.closed = True
        if MLX_AVAILABLE and self.backend == "mlx":
            mx.clear_cache()

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def _require_open(self) -> None:
        if self.closed:
            raise R2MapProtocolError("checkpoint registry is closed")


def score_grouped_request(
    registry: R2MapCheckpointRegistry,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray[Any, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    """Validate and exhaustively score one request, partitioned only by checkpoint."""
    groups, offsets = validate_grouped_request(metadata, tensors)
    total_candidates = int(offsets[-1])
    outputs = _allocate_response_tensors(total_candidates)
    checkpoint_waves: OrderedDict[str, list[int]] = OrderedDict()
    for group_index, group in enumerate(groups):
        checkpoint_waves.setdefault(group["model"]["checkpoint_id"], []).append(group_index)

    registry.begin_wave()
    try:
        for checkpoint_id, group_indices in checkpoint_waves.items():
            identity = groups[group_indices[0]]["model"]
            entry = registry.get(identity)
            if any(groups[index]["model"] != identity for index in group_indices):
                raise R2MapProtocolError(
                    f"checkpoint cohort {checkpoint_id} contains inconsistent model hashes"
                )
            if getattr(entry.model, "array_backend", "mlx") == "numpy":
                for group_index in group_indices:
                    start, stop = int(offsets[group_index]), int(offsets[group_index + 1])
                    prediction = entry.model.score_actions(
                        _numpy_public_state(tensors, "parent", slice(group_index, group_index + 1)),
                        _numpy_public_state(tensors, "candidate", slice(start, stop)),
                        tensors["action_bytes"][start:stop],
                        tensors["exact_afterstate_scores"][start:stop],
                    )
                    outputs["action_scores"][start:stop] = prediction.action_scores
                    outputs["predicted_score_to_go"][start:stop] = prediction.predicted_score_to_go
                    outputs["predicted_score_components_to_go"][start:stop] = (
                        prediction.predicted_score_components_to_go
                    )
                    outputs["bootstrap_policy_logits"][start:stop] = (
                        prediction.bootstrap_policy_logits
                    )
            else:
                batch = _materialize_wave_batch(tensors, offsets, group_indices)
                prediction = entry.model.score_actions(batch)
                _evaluate_prediction(prediction)
                for local_index, group_index in enumerate(group_indices):
                    global_start = int(offsets[group_index])
                    global_stop = int(offsets[group_index + 1])
                    count = global_stop - global_start
                    _copy_group_prediction(
                        outputs,
                        prediction,
                        local_index=local_index,
                        global_start=global_start,
                        global_stop=global_stop,
                        candidate_count=count,
                    )
    finally:
        registry.end_wave()

    response_groups = []
    for index, group in enumerate(groups):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        response_groups.append(
            {
                "group_id": group["group_id"],
                "decision_id": group["decision_id"],
                "model": group["model"],
                "candidate_offset": start,
                "candidate_count": stop - start,
                "action_ids": group["action_ids"],
                "ordered_action_ids_blake3": group["ordered_action_ids_blake3"],
                "diagnostics": {
                    "parent_groups_encoded": 1,
                    "actions_enumerated": stop - start,
                    "actions_scored": stop - start,
                    "complete_cardinality": True,
                },
            }
        )
    response_metadata = {
        "schema_version": 1,
        "schema_id": RESPONSE_SCHEMA,
        "request_schema_blake3": REQUEST_SCHEMA_BLAKE3,
        "request_identity_blake3": _request_identity_blake3(metadata),
        "group_count": len(groups),
        "candidate_count": total_candidates,
        "groups": response_groups,
        "diagnostics": {
            "reference_exhaustive": True,
            "pruned_actions": 0,
            "quantized": False,
            "remote_inference": False,
            "parent_encoder_calls": len(checkpoint_waves),
            "checkpoint_waves": len(checkpoint_waves),
        },
    }
    return response_metadata, outputs


def validate_grouped_request(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray[Any, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray[Any, np.dtype[np.int32]]]:
    if (
        metadata.get("schema_version") != 1
        or metadata.get("schema_id") != REQUEST_SCHEMA
        or metadata.get("request_schema_blake3") != REQUEST_SCHEMA_BLAKE3
    ):
        raise R2MapProtocolError("grouped request schema or schema hash differs")
    groups_value = metadata.get("groups")
    if not isinstance(groups_value, list):
        raise R2MapProtocolError("grouped request groups must be an array")
    groups = [dict(group) if isinstance(group, dict) else {} for group in groups_value]
    if not 1 <= len(groups) <= MAX_GROUPS or metadata.get("group_count") != len(groups):
        raise R2MapProtocolError("grouped request group cardinality differs")
    if set(tensors) != set(REQUEST_TENSOR_DTYPES):
        raise R2MapProtocolError("grouped request tensor names differ")
    offsets = tensors["candidate_offsets"]
    if offsets.dtype != np.dtype("<i4") or offsets.shape != (len(groups) + 1,):
        raise R2MapProtocolError("candidate offsets shape or dtype differs")
    if offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]):
        raise R2MapProtocolError("candidate offsets must be strictly increasing from zero")
    total_candidates = int(offsets[-1])
    if metadata.get("candidate_count") != total_candidates or not (
        1 <= total_candidates <= MAX_TOTAL_CANDIDATES
    ):
        raise R2MapProtocolError("grouped request total candidate count differs")
    validate_action_groups(groups, offsets)
    _validate_request_tensor_shapes(tensors, len(groups), total_candidates)
    return groups, offsets


def validate_action_groups(
    groups: Sequence[Mapping[str, Any]],
    offsets: Sequence[int] | np.ndarray[Any, Any],
) -> None:
    seen_groups: set[str] = set()
    for index, group in enumerate(groups):
        required = {
            "group_id",
            "decision_id",
            "model",
            "expected_legal_action_count",
            "action_ids",
            "enumeration_indices",
            "ordered_action_ids_blake3",
        }
        if set(group) != required:
            raise R2MapProtocolError(f"group {index} field set differs")
        group_id = group["group_id"]
        decision_id = group["decision_id"]
        if not _is_blake3(group_id) or not _is_blake3(decision_id):
            raise R2MapProtocolError(f"group {index} identity must be a BLAKE3 digest")
        if group_id in seen_groups:
            raise R2MapProtocolError("grouped request contains a duplicate group identity")
        seen_groups.add(group_id)
        _validate_model_identity(group["model"])
        count = int(offsets[index + 1]) - int(offsets[index])
        if not 1 <= count <= PROTOCOL_MAX_CANDIDATES_PER_GROUP:
            raise R2MapProtocolError(f"group {index} candidate count exceeds protocol ceiling")
        if group["expected_legal_action_count"] != count:
            raise R2MapProtocolError(f"group {index} is a partial legal-action set")
        action_ids = group["action_ids"]
        indices = group["enumeration_indices"]
        if (
            not isinstance(action_ids, list)
            or len(action_ids) != count
            or not all(_is_blake3(action_id) for action_id in action_ids)
        ):
            raise R2MapProtocolError(f"group {index} action identity cardinality differs")
        if len(set(action_ids)) != count:
            raise R2MapProtocolError(f"group {index} contains a duplicate action identity")
        if indices != list(range(count)):
            raise R2MapProtocolError(f"group {index} action enumeration was reordered")
        expected_digest = _ordered_action_ids_blake3(action_ids)
        if group["ordered_action_ids_blake3"] != expected_digest:
            raise R2MapProtocolError(f"group {index} ordered action digest differs")


def score_market_decision_request(
    registry: R2MapCheckpointRegistry,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray[Any, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    """Score complete public pre-refill action screens without future leakage."""
    groups, offsets = validate_market_decision_request(metadata, tensors)
    total_actions = int(offsets[-1])
    outputs = {
        name: np.zeros((total_actions,), dtype=dtype)
        for name, dtype in MARKET_RESPONSE_TENSOR_DTYPES.items()
    }
    waves: OrderedDict[str, list[int]] = OrderedDict()
    for index, group in enumerate(groups):
        waves.setdefault(group["model"]["checkpoint_id"], []).append(index)
    registry.begin_wave()
    try:
        for checkpoint_id, indices in waves.items():
            identity = groups[indices[0]]["model"]
            if any(groups[index]["model"] != identity for index in indices):
                raise R2MapProtocolError(
                    f"market checkpoint cohort {checkpoint_id} has inconsistent hashes"
                )
            entry = registry.get(identity)
            if getattr(entry.model, "array_backend", "mlx") == "numpy":
                for group_index in indices:
                    start, stop = int(offsets[group_index]), int(offsets[group_index + 1])
                    prediction = entry.model.score_market_decisions(
                        _numpy_public_state(tensors, "parent", slice(group_index, group_index + 1)),
                        tensors["action_bytes"][start:stop],
                        float(tensors["exact_current_scores"][group_index]),
                    )
                    outputs["market_action_scores"][start:stop] = prediction.action_scores
                    outputs["market_predicted_score_to_go"][start:stop] = (
                        prediction.predicted_score_to_go
                    )
            else:
                batch = _materialize_market_wave_batch(tensors, offsets, indices)
                prediction = entry.model.score_market_decisions(batch)
                mx.eval(
                    prediction.action_scores,
                    prediction.predicted_score_to_go,
                )
                for local, group_index in enumerate(indices):
                    start = int(offsets[group_index])
                    stop = int(offsets[group_index + 1])
                    count = stop - start
                    outputs["market_action_scores"][start:stop] = np.asarray(
                        prediction.action_scores[local, :count]
                    )
                    outputs["market_predicted_score_to_go"][start:stop] = np.asarray(
                        prediction.predicted_score_to_go[local, :count]
                    )
    finally:
        registry.end_wave()
    response_groups = []
    for index, group in enumerate(groups):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        response_groups.append(
            {
                "group_id": group["group_id"],
                "decision_id": group["decision_id"],
                "model": group["model"],
                "action_offset": start,
                "action_count": stop - start,
                "action_ids": group["action_ids"],
                "ordered_action_ids_blake3": group["ordered_action_ids_blake3"],
                "decision_kind": group["decision_kind"],
                "public_nature_tokens": group["public_nature_tokens"],
                "public_wildlife_bag_total": group["public_wildlife_bag_total"],
                "public_wildlife_bag_counts": group["public_wildlife_bag_counts"],
                "public_market_wildlife": group["public_market_wildlife"],
                "diagnostics": {
                    "actions_enumerated": stop - start,
                    "actions_scored": stop - start,
                    "complete_cardinality": True,
                    "hidden_refill_inputs": 0,
                },
            }
        )
    return (
        {
            "schema_version": 1,
            "schema_id": MARKET_RESPONSE_SCHEMA,
            "request_schema_blake3": MARKET_REQUEST_SCHEMA_BLAKE3,
            "response_schema_blake3": MARKET_RESPONSE_SCHEMA_BLAKE3,
            "request_identity_blake3": _request_identity_blake3(metadata),
            "group_count": len(groups),
            "action_count": total_actions,
            "groups": response_groups,
            "diagnostics": {
                "reference_exhaustive": True,
                "pruned_actions": 0,
                "future_refill_tensors": 0,
                "checkpoint_waves": len(waves),
            },
        },
        outputs,
    )


def validate_market_decision_request(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray[Any, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray[Any, np.dtype[np.int32]]]:
    if (
        metadata.get("schema_version") != 1
        or metadata.get("schema_id") != MARKET_REQUEST_SCHEMA
        or metadata.get("request_schema_blake3") != MARKET_REQUEST_SCHEMA_BLAKE3
    ):
        raise R2MapProtocolError("market decision request schema or hash differs")
    raw_groups = metadata.get("groups")
    if not isinstance(raw_groups, list):
        raise R2MapProtocolError("market decision groups must be an array")
    groups = [dict(group) if isinstance(group, dict) else {} for group in raw_groups]
    if not 1 <= len(groups) <= MAX_GROUPS or metadata.get("group_count") != len(groups):
        raise R2MapProtocolError("market decision group cardinality differs")
    if set(tensors) != set(MARKET_REQUEST_TENSOR_DTYPES):
        raise R2MapProtocolError("market decision tensor names differ")
    offsets = tensors["action_offsets"]
    if offsets.dtype != np.dtype("<i4") or offsets.shape != (len(groups) + 1,):
        raise R2MapProtocolError("market action offsets shape or dtype differs")
    if offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]):
        raise R2MapProtocolError("market action offsets must increase from zero")
    total = int(offsets[-1])
    if metadata.get("action_count") != total or not 1 <= total <= MAX_GROUPS * 16:
        raise R2MapProtocolError("market decision total action count differs")
    base_groups = []
    for index, group in enumerate(groups):
        if set(group) != {
            "group_id",
            "decision_id",
            "model",
            "expected_legal_action_count",
            "action_ids",
            "enumeration_indices",
            "ordered_action_ids_blake3",
            "public_nature_tokens",
            "public_wildlife_bag_total",
            "public_wildlife_bag_counts",
            "public_market_wildlife",
            "decision_kind",
        }:
            raise R2MapProtocolError(f"market group {index} field set differs")
        tokens = group["public_nature_tokens"]
        bag_total = group["public_wildlife_bag_total"]
        bag_counts = group["public_wildlife_bag_counts"]
        market_wildlife = group["public_market_wildlife"]
        kind = group["decision_kind"]
        if (
            not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or not 0 <= tokens <= 255
            or not isinstance(bag_total, int)
            or isinstance(bag_total, bool)
            or not 0 <= bag_total <= 100
            or not isinstance(bag_counts, list)
            or len(bag_counts) != 5
            or any(
                not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 20
                for value in bag_counts
            )
            or sum(bag_counts) != bag_total
            or not isinstance(market_wildlife, list)
            or len(market_wildlife) != 4
            or any(
                not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4
                for value in market_wildlife
            )
            or kind not in {0, 1}
        ):
            raise R2MapProtocolError(f"market group {index} public resource metadata differs")
        base_groups.append(
            {
                name: value
                for name, value in group.items()
                if name
                not in {
                    "public_nature_tokens",
                    "public_wildlife_bag_total",
                    "public_wildlife_bag_counts",
                    "public_market_wildlife",
                    "decision_kind",
                }
            }
        )
    validate_action_groups(base_groups, offsets)
    _validate_market_tensor_shapes(tensors, len(groups), total)
    raw_actions = tensors["action_bytes"]
    for index, group in enumerate(groups):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        validate_canonical_market_action_order(
            raw_actions[start:stop],
            decision_kind=MarketDecisionKind(group["decision_kind"]),
            public_nature_tokens=group["public_nature_tokens"],
            public_wildlife_bag_total=group["public_wildlife_bag_total"],
            public_wildlife_bag_counts=group["public_wildlife_bag_counts"],
            public_market_wildlife=group["public_market_wildlife"],
        )
        expected_ids = [
            market_decision_action_id(group["decision_id"], row.tobytes())
            for row in raw_actions[start:stop]
        ]
        if group["action_ids"] != expected_ids:
            raise R2MapProtocolError("market action identities differ from canonical bytes")
    return groups, offsets


def encode_tensor_frame(
    *,
    message_type: int,
    request_id: int,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray[Any, Any]],
    expected_dtypes: Mapping[str, str],
) -> bytes:
    if not 0 <= request_id <= 0xFFFFFFFF:
        raise R2MapProtocolError("request id is outside uint32")
    if set(tensors) != set(expected_dtypes):
        raise R2MapProtocolError("tensor frame names differ")
    payload = bytearray()
    descriptors = []
    for name in expected_dtypes:
        value = np.ascontiguousarray(tensors[name])
        if value.dtype != np.dtype(expected_dtypes[name]):
            raise R2MapProtocolError(f"tensor {name} dtype differs")
        encoded = value.tobytes(order="C")
        descriptors.append(
            {
                "name": name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "offset": len(payload),
                "bytes": len(encoded),
                "blake3": blake3.blake3(encoded).hexdigest(),
            }
        )
        payload.extend(encoded)
    envelope = dict(metadata)
    envelope["tensors"] = descriptors
    envelope["tensor_payload_blake3"] = blake3.blake3(payload).hexdigest()
    metadata_bytes = _canonical_json(envelope)
    if len(metadata_bytes) > MAX_METADATA_BYTES or len(payload) > MAX_TENSOR_BYTES:
        raise R2MapProtocolError("tensor frame exceeds the reference size ceiling")
    return (
        FRAME_HEADER.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            message_type,
            request_id,
            len(metadata_bytes),
            len(payload),
        )
        + metadata_bytes
        + payload
    )


def decode_tensor_payload(
    metadata_bytes: bytes,
    payload: bytes,
    *,
    expected_dtypes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    try:
        metadata = json.loads(metadata_bytes)
    except json.JSONDecodeError as error:
        raise R2MapProtocolError(f"invalid tensor frame metadata: {error}") from error
    if not isinstance(metadata, dict):
        raise R2MapProtocolError("tensor frame metadata must be an object")
    descriptors = metadata.pop("tensors", None)
    claimed_payload_hash = metadata.pop("tensor_payload_blake3", None)
    if claimed_payload_hash != blake3.blake3(payload).hexdigest():
        raise R2MapProtocolError("tensor frame payload hash differs")
    if not isinstance(descriptors, list) or len(descriptors) != len(expected_dtypes):
        raise R2MapProtocolError("tensor frame descriptor count differs")
    tensors: dict[str, np.ndarray[Any, Any]] = {}
    next_offset = 0
    for descriptor, (expected_name, expected_dtype) in zip(
        descriptors, expected_dtypes.items(), strict=True
    ):
        if not isinstance(descriptor, dict) or descriptor.get("name") != expected_name:
            raise R2MapProtocolError("tensor frame descriptor order differs")
        if descriptor.get("dtype") != np.dtype(expected_dtype).str:
            raise R2MapProtocolError(f"tensor {expected_name} descriptor dtype differs")
        if descriptor.get("offset") != next_offset:
            raise R2MapProtocolError("tensor frame contains a gap, overlap, or reordering")
        size = descriptor.get("bytes")
        shape = descriptor.get("shape")
        if (
            not isinstance(size, int)
            or size < 0
            or not isinstance(shape, list)
            or not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape)
        ):
            raise R2MapProtocolError(f"tensor {expected_name} descriptor is malformed")
        stop = next_offset + size
        encoded = payload[next_offset:stop]
        if len(encoded) != size or descriptor.get("blake3") != blake3.blake3(encoded).hexdigest():
            raise R2MapProtocolError(f"tensor {expected_name} bytes failed integrity")
        dtype = np.dtype(expected_dtype)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected_bytes != size:
            raise R2MapProtocolError(f"tensor {expected_name} shape and byte count differ")
        tensors[expected_name] = np.frombuffer(encoded, dtype=dtype).reshape(shape).copy()
        next_offset = stop
    if next_offset != len(payload):
        raise R2MapProtocolError("tensor frame contains trailing payload bytes")
    return metadata, tensors


def serve_r2_map(
    registry: R2MapCheckpointRegistry,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
) -> None:
    """Serve complete grouped requests until clean shutdown, EOF, or a fatal frame error."""
    try:
        while True:
            header = _read_exact_or_eof(input_stream, FRAME_HEADER.size)
            if header is None:
                return
            magic, version, message_type, request_id, metadata_size, tensor_size = (
                FRAME_HEADER.unpack(header)
            )
            try:
                if magic != PROTOCOL_MAGIC or version != PROTOCOL_VERSION:
                    raise R2MapProtocolError("incompatible R2-MAP protocol header")
                if metadata_size > MAX_METADATA_BYTES or tensor_size > MAX_TENSOR_BYTES:
                    raise R2MapProtocolError("R2-MAP frame length exceeds protocol ceiling")
                if message_type == MESSAGE_SHUTDOWN:
                    if metadata_size or tensor_size:
                        raise R2MapProtocolError("shutdown frame cannot contain a payload")
                    return
                if message_type not in {MESSAGE_SCORE_GROUPS, MESSAGE_SCORE_MARKET_DECISIONS}:
                    raise R2MapProtocolError(f"unsupported R2-MAP message type {message_type}")
                metadata_bytes = _read_exact(input_stream, metadata_size)
                payload = _read_exact(input_stream, tensor_size)
                request_dtypes = (
                    REQUEST_TENSOR_DTYPES
                    if message_type == MESSAGE_SCORE_GROUPS
                    else MARKET_REQUEST_TENSOR_DTYPES
                )
                metadata, tensors = decode_tensor_payload(
                    metadata_bytes,
                    payload,
                    expected_dtypes=request_dtypes,
                )
                if message_type == MESSAGE_SCORE_GROUPS:
                    response_metadata, response_tensors = score_grouped_request(
                        registry, metadata, tensors
                    )
                    response_type = MESSAGE_SCORE_RESPONSE
                    response_dtypes = RESPONSE_TENSOR_DTYPES
                else:
                    response_metadata, response_tensors = score_market_decision_request(
                        registry, metadata, tensors
                    )
                    response_type = MESSAGE_SCORE_MARKET_DECISIONS_RESPONSE
                    response_dtypes = MARKET_RESPONSE_TENSOR_DTYPES
                output_stream.write(
                    encode_tensor_frame(
                        message_type=response_type,
                        request_id=request_id,
                        metadata=response_metadata,
                        tensors=response_tensors,
                        expected_dtypes=response_dtypes,
                    )
                )
                output_stream.flush()
            except Exception as error:
                encoded = str(error).encode()
                output_stream.write(
                    FRAME_HEADER.pack(
                        PROTOCOL_MAGIC,
                        PROTOCOL_VERSION,
                        MESSAGE_ERROR,
                        request_id,
                        0,
                        len(encoded),
                    )
                )
                output_stream.write(encoded)
                output_stream.flush()
                return
    finally:
        registry.close()


def _numpy_public_state(
    tensors: Mapping[str, np.ndarray[Any, Any]], prefix: str, rows: slice
) -> dict[str, np.ndarray[Any, Any]]:
    return {
        "token_features": np.asarray(tensors[f"{prefix}_token_features"][rows], dtype=np.float32),
        "token_types": np.asarray(tensors[f"{prefix}_token_types"][rows], dtype=np.int32),
        "token_mask": np.asarray(tensors[f"{prefix}_token_mask"][rows], dtype=np.bool_),
        "market_features": np.asarray(tensors[f"{prefix}_market_features"][rows], dtype=np.float32),
        "market_mask": np.asarray(tensors[f"{prefix}_market_mask"][rows], dtype=np.bool_),
        "player_features": np.asarray(tensors[f"{prefix}_player_features"][rows], dtype=np.float32),
        "player_mask": np.asarray(tensors[f"{prefix}_player_mask"][rows], dtype=np.bool_),
        "global_features": np.asarray(tensors[f"{prefix}_global_features"][rows], dtype=np.float32),
    }


def _materialize_wave_batch(
    tensors: Mapping[str, np.ndarray[Any, Any]],
    offsets: np.ndarray[Any, Any],
    group_indices: Sequence[int],
) -> R2MapBatch:
    counts = [int(offsets[index + 1] - offsets[index]) for index in group_indices]
    maximum = max(counts)

    # The overwhelmingly common reference-serving case is one complete legal
    # set. Preserve its contiguous payload views instead of duplicating every
    # padded afterstate into a second NumPy allocation before MLX ingestion.
    # This is still one exhaustive model call: no chunking, pruning, or staged
    # selection is introduced.
    single_group = len(group_indices) == 1
    single_start = int(offsets[group_indices[0]]) if single_group else 0
    single_stop = int(offsets[group_indices[0] + 1]) if single_group else 0

    def parents(name: str) -> np.ndarray[Any, Any]:
        return np.ascontiguousarray(tensors[f"parent_{name}"][list(group_indices)])

    def candidates(name: str) -> np.ndarray[Any, Any]:
        source = tensors[f"candidate_{name}"]
        if single_group:
            return source[single_start:single_stop][None, ...]
        result = np.zeros((len(group_indices), maximum, *source.shape[1:]), dtype=source.dtype)
        for local, group in enumerate(group_indices):
            start = int(offsets[group])
            stop = int(offsets[group + 1])
            result[local, : stop - start] = source[start:stop]
        return result

    candidate_mask = np.zeros((len(group_indices), maximum), dtype=np.bool_)
    action_features = np.zeros((len(group_indices), maximum, 140), dtype=np.float32)
    exact_scores = np.zeros((len(group_indices), maximum), dtype=np.float32)
    for local, group in enumerate(group_indices):
        start = int(offsets[group])
        stop = int(offsets[group + 1])
        count = stop - start
        candidate_mask[local, :count] = True
        action_features[local, :count] = decode_action_features(tensors["action_bytes"][start:stop])
        exact_scores[local, :count] = tensors["exact_afterstate_scores"][start:stop]
    batch = R2MapBatch(
        parent=R2MapPublicState(
            token_features=mx.array(parents("token_features")),
            token_types=mx.array(parents("token_types")),
            token_mask=mx.array(parents("token_mask").astype(np.bool_)),
            market_features=mx.array(parents("market_features")),
            market_mask=mx.array(parents("market_mask").astype(np.bool_)),
            player_features=mx.array(parents("player_features")),
            player_mask=mx.array(parents("player_mask").astype(np.bool_)),
            global_features=mx.array(parents("global_features")),
        ),
        candidates=R2MapPublicState(
            token_features=mx.array(candidates("token_features")),
            token_types=mx.array(candidates("token_types")),
            token_mask=mx.array(candidates("token_mask").astype(np.bool_)),
            market_features=mx.array(candidates("market_features")),
            market_mask=mx.array(candidates("market_mask").astype(np.bool_)),
            player_features=mx.array(candidates("player_features")),
            player_mask=mx.array(candidates("player_mask").astype(np.bool_)),
            global_features=mx.array(candidates("global_features")),
        ),
        candidate_mask=mx.array(candidate_mask),
        action_features=mx.array(action_features),
        exact_afterstate_scores=mx.array(exact_scores),
    )
    batch.validate()
    return batch


def _materialize_market_wave_batch(
    tensors: Mapping[str, np.ndarray[Any, Any]],
    offsets: np.ndarray[Any, Any],
    group_indices: Sequence[int],
) -> R2MapMarketDecisionBatch:
    counts = [int(offsets[index + 1] - offsets[index]) for index in group_indices]
    maximum = max(counts)

    def parents(name: str) -> np.ndarray[Any, Any]:
        return np.ascontiguousarray(tensors[f"parent_{name}"][list(group_indices)])

    mask = np.zeros((len(group_indices), maximum), dtype=np.bool_)
    features = np.zeros((len(group_indices), maximum, 16), dtype=np.float32)
    for local, group in enumerate(group_indices):
        start, stop = int(offsets[group]), int(offsets[group + 1])
        count = stop - start
        mask[local, :count] = True
        features[local, :count] = decode_market_action_features(tensors["action_bytes"][start:stop])
    batch = R2MapMarketDecisionBatch(
        public_state=R2MapPublicState(
            token_features=mx.array(parents("token_features")),
            token_types=mx.array(parents("token_types")),
            token_mask=mx.array(parents("token_mask").astype(np.bool_)),
            market_features=mx.array(parents("market_features")),
            market_mask=mx.array(parents("market_mask").astype(np.bool_)),
            player_features=mx.array(parents("player_features")),
            player_mask=mx.array(parents("player_mask").astype(np.bool_)),
            global_features=mx.array(parents("global_features")),
        ),
        action_mask=mx.array(mask),
        action_features=mx.array(features),
        exact_current_scores=mx.array(tensors["exact_current_scores"][list(group_indices)]),
    )
    batch.validate()
    return batch


def _validate_request_tensor_shapes(
    tensors: Mapping[str, np.ndarray[Any, Any]], groups: int, candidates: int
) -> None:
    expected = {
        "candidate_offsets": (groups + 1,),
        "parent_token_features": (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES),
        "parent_token_types": (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "parent_token_mask": (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "parent_market_features": (groups, 4, MARKET_FEATURES),
        "parent_market_mask": (groups, 4),
        "parent_player_features": (groups, BOARD_SLOTS, PLAYER_FEATURES),
        "parent_player_mask": (groups, BOARD_SLOTS),
        "parent_global_features": (groups, GLOBAL_FEATURES),
        "candidate_token_features": (
            candidates,
            BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            TOKEN_FEATURES,
        ),
        "candidate_token_types": (candidates, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "candidate_token_mask": (candidates, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "candidate_market_features": (candidates, 4, MARKET_FEATURES),
        "candidate_market_mask": (candidates, 4),
        "candidate_player_features": (candidates, BOARD_SLOTS, PLAYER_FEATURES),
        "candidate_player_mask": (candidates, BOARD_SLOTS),
        "candidate_global_features": (candidates, GLOBAL_FEATURES),
        "action_bytes": (candidates, GRADED_ORACLE_ACTION_FEATURE_SIZE),
        "exact_afterstate_scores": (candidates,),
    }
    for name, shape in expected.items():
        value = tensors[name]
        if value.dtype != np.dtype(REQUEST_TENSOR_DTYPES[name]) or value.shape != shape:
            raise R2MapProtocolError(f"request tensor {name} shape or dtype differs")


def _validate_market_tensor_shapes(
    tensors: Mapping[str, np.ndarray[Any, Any]], groups: int, actions: int
) -> None:
    expected = {
        "action_offsets": (groups + 1,),
        "parent_token_features": (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY, TOKEN_FEATURES),
        "parent_token_types": (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "parent_token_mask": (groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY),
        "parent_market_features": (groups, 4, MARKET_FEATURES),
        "parent_market_mask": (groups, 4),
        "parent_player_features": (groups, BOARD_SLOTS, PLAYER_FEATURES),
        "parent_player_mask": (groups, BOARD_SLOTS),
        "parent_global_features": (groups, GLOBAL_FEATURES),
        "action_bytes": (actions, MARKET_DECISION_ACTION_SIZE),
        "exact_current_scores": (groups,),
    }
    for name, shape in expected.items():
        value = tensors[name]
        if value.dtype != np.dtype(MARKET_REQUEST_TENSOR_DTYPES[name]) or value.shape != shape:
            raise R2MapProtocolError(f"market request tensor {name} shape or dtype differs")


def _allocate_response_tensors(candidates: int) -> dict[str, np.ndarray[Any, Any]]:
    shapes = {
        "action_scores": (candidates,),
        "predicted_score_to_go": (candidates,),
        "predicted_score_components_to_go": (candidates, 11),
        "bootstrap_policy_logits": (candidates,),
    }
    return {
        name: np.zeros(shape, dtype=dtype)
        for (name, dtype), shape in zip(
            RESPONSE_TENSOR_DTYPES.items(), shapes.values(), strict=True
        )
    }


def _copy_group_prediction(
    outputs: dict[str, np.ndarray[Any, Any]],
    prediction: Any,
    *,
    local_index: int,
    global_start: int,
    global_stop: int,
    candidate_count: int,
) -> None:
    outputs["action_scores"][global_start:global_stop] = np.asarray(
        prediction.action_scores[local_index, :candidate_count]
    )
    outputs["predicted_score_to_go"][global_start:global_stop] = np.asarray(
        prediction.predicted_score_to_go[local_index, :candidate_count]
    )
    outputs["predicted_score_components_to_go"][global_start:global_stop] = np.asarray(
        prediction.predicted_score_components_to_go[local_index, :candidate_count]
    )
    outputs["bootstrap_policy_logits"][global_start:global_stop] = np.asarray(
        prediction.bootstrap_policy_logits[local_index, :candidate_count]
    )


def _evaluate_prediction(prediction: Any) -> None:
    mx.eval(
        prediction.action_scores,
        prediction.predicted_score_to_go,
        prediction.predicted_score_components_to_go,
        prediction.bootstrap_policy_logits,
    )
    raw_mask = np.asarray(prediction.candidate_mask)
    if raw_mask.dtype != np.dtype(np.bool_) or raw_mask.ndim != 2:
        raise R2MapProtocolError("prediction candidate mask shape or dtype differs")
    mask = raw_mask
    groups, candidates = mask.shape
    candidate_shapes = {
        "action_scores": (groups, candidates),
        "predicted_score_to_go": (groups, candidates),
        "predicted_score_components_to_go": (groups, candidates, 11),
        "bootstrap_policy_logits": (groups, candidates),
    }
    for name, shape in candidate_shapes.items():
        value = np.asarray(getattr(prediction, name))
        if value.shape != shape:
            raise R2MapProtocolError(f"prediction tensor {name} shape differs")
        if not np.all(np.isfinite(value[mask])):
            raise R2MapProtocolError(f"prediction tensor {name} is non-finite on a legal action")


def _validate_model_identity(value: Any) -> None:
    required = {
        "checkpoint_id",
        "checkpoint_manifest_blake3",
        "model_config_blake3",
        "model_weights_blake3",
        "verification_id",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise R2MapProtocolError("model identity field set differs")
    checkpoint_id = value["checkpoint_id"]
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise R2MapProtocolError("model checkpoint id is empty")
    for name in required - {"checkpoint_id"}:
        if not _is_blake3(value[name]):
            raise R2MapProtocolError(f"model identity {name} must be a BLAKE3 digest")


def _validate_protocol_identity(value: Any) -> dict[str, list[int]]:
    required = {"collector_hash", "source_hash", "serving_protocol_hash"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise R2MapProtocolError("serving protocol identity field set differs")
    result: dict[str, list[int]] = {}
    for name in sorted(required):
        raw = value[name]
        if (
            not isinstance(raw, list)
            or len(raw) != 32
            or any(
                not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 255
                for item in raw
            )
            or raw == [0] * 32
        ):
            raise R2MapProtocolError(f"serving protocol identity {name} is invalid")
        result[name] = list(raw)
    return result


def _ordered_action_ids_blake3(action_ids: Sequence[str]) -> str:
    return blake3.blake3(_canonical_json(list(action_ids))).hexdigest()


def ordered_action_ids_blake3(action_ids: Sequence[str]) -> str:
    """Public protocol helper shared with the future Rust request encoder."""
    return _ordered_action_ids_blake3(action_ids)


def _request_identity_blake3(metadata: Mapping[str, Any]) -> str:
    value = dict(metadata)
    value.pop("tensors", None)
    value.pop("tensor_payload_blake3", None)
    return blake3.blake3(_canonical_json(value)).hexdigest()


def _is_blake3(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _file_blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_portable_checkpoint(
    run_dir: Path, checkpoint_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the immutable serving surface without importing MLX.

    Rust verifies the complete bundle before launching the service.  This
    independent boundary rechecks the model/config/checkpoint/receipt bindings
    most relevant to a Linux process before mapping any tensor bytes.
    """
    try:
        manifest = json.loads((checkpoint_path / "checkpoint.json").read_text())
        receipt = json.loads(
            (run_dir / "verifications" / f"{checkpoint_path.name}.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise R2MapProtocolError(f"cannot read portable checkpoint metadata: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 2
        or manifest.get("schema_id") != "r2-map-checkpoint-v2"
        or manifest.get("checkpoint_id") != checkpoint_path.name
        or not isinstance(manifest.get("model_config"), dict)
        or not isinstance(manifest.get("identity"), dict)
        or not isinstance(manifest.get("files"), dict)
    ):
        raise R2MapProtocolError("portable checkpoint manifest schema differs")
    descriptor = manifest["files"].get("model.safetensors")
    model_path = checkpoint_path / "model.safetensors"
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("bytes") != model_path.stat().st_size
        or not _is_blake3(descriptor.get("blake3"))
        or _file_blake3(model_path) != descriptor["blake3"]
        or manifest["identity"].get("model_config_blake3")
        != blake3.blake3(_canonical_json(manifest["model_config"])).hexdigest()
    ):
        raise R2MapProtocolError("portable checkpoint model or config hash differs")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 2
        or receipt.get("schema_id") != "r2-map-checkpoint-verification-v2"
        or receipt.get("checkpoint_id") != checkpoint_path.name
        or receipt.get("checkpoint_manifest_blake3")
        != _file_blake3(checkpoint_path / "checkpoint.json")
        or receipt.get("exact_prediction_match") is not True
        or receipt.get("exact_next_batch_match") is not True
        or not _is_blake3(receipt.get("verification_id"))
    ):
        raise R2MapProtocolError("portable checkpoint verification receipt differs")
    return manifest, receipt


def _read_exact_or_eof(stream: BinaryIO, size: int) -> bytes | None:
    first = stream.read(size)
    if not first:
        return None
    if len(first) == size:
        return first
    return first + _read_exact(stream, size - len(first))


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("R2-MAP stream ended inside a frame")
        chunks.extend(chunk)
    return bytes(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--pin", action="append", default=[])
    parser.add_argument("--registry-capacity", type=int, default=DEFAULT_REGISTRY_CAPACITY)
    parser.add_argument("--backend", choices=("auto", "mlx", "numpy"), default="auto")
    parser.add_argument("--candidate-chunk-size", type=int, default=DEFAULT_CANDIDATE_CHUNK_SIZE)
    arguments = parser.parse_args()
    registry = R2MapCheckpointRegistry(
        arguments.registry_capacity,
        backend=arguments.backend,
        candidate_chunk_size=arguments.candidate_chunk_size,
    )
    if arguments.bundle is not None:
        if arguments.run_dir is not None or arguments.checkpoint or arguments.pin:
            parser.error("--bundle cannot be combined with direct checkpoint arguments")
        registry.register_verified_bundle(arguments.bundle)
    else:
        if arguments.run_dir is None or not arguments.checkpoint:
            parser.error("provide --bundle or both --run-dir and --checkpoint")
        for checkpoint in arguments.checkpoint:
            registry.register_verified_checkpoint(
                run_dir=arguments.run_dir,
                checkpoint_path=checkpoint,
                pinned=checkpoint.name in set(arguments.pin),
            )
    device = str(mx.default_device()) if registry.backend == "mlx" else "numpy-cpu"
    print(
        f"R2-MAP serving {len(registry.checkpoint_ids)} local checkpoints on {device}",
        file=sys.stderr,
    )
    serve_r2_map(registry, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    main()
