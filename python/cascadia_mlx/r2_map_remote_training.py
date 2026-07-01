"""Receipt-preserving John1 training I/O over frozen John2 storage.

This module is the production boundary between the filesystem-free MLX
trainer and the authoritative John2 campaign root. It never creates a local
dataset window, checkpoint, loss stream, pointer, cache, or temporary file.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import blake3

from cascadia_mlx.checkpoint import (
    R2MapCheckpointBundle,
    build_r2_map_checkpoint_pointer_document,
    validate_r2_map_checkpoint_pointer_document,
    verify_r2_map_checkpoint_bundle,
)
from cascadia_mlx.r2_map_dataset import (
    MAX_IN_MEMORY_STREAM_BYTES,
    validate_compact_index_value,
)
from cascadia_mlx.r2_map_remote_storage import (
    REMOTE_ROOT,
    RemoteOperationError,
    RemoteProtocolError,
    RemoteStorageClient,
    TransactionObject,
    build_transaction_manifest,
    canonical_json,
    content_sha256,
)
from cascadia_mlx.r2_map_verify import validate_verification_receipt_value

MAX_COMPACT_INDEX_BYTES = 64 << 20
MAX_CHECKPOINT_OBJECT_BYTES = 1 << 30
MAX_CHECKPOINT_BUNDLE_BYTES = 2 << 30
MAX_VERIFICATION_RECEIPT_BYTES = 2 << 20
REMOTE_WINDOW_READ_BYTES = 64 << 20
REMOTE_TRAINING_SCHEMA = "cascadia.r2-map.remote-training-publication.v1"


class R2MapRemoteTrainingError(RuntimeError):
    """Frozen remote training I/O or its evidence failed closed."""


def _canonical_blake3(value: Any) -> str:
    return blake3.blake3(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class RemoteObjectEvidence:
    relative: str
    object_token: dict[str, Any]
    open_receipt: dict[str, str]
    range_receipts: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative": self.relative,
            "object_token": dict(self.object_token),
            "open_receipt": dict(self.open_receipt),
            "range_receipts": [dict(value) for value in self.range_receipts],
        }


@dataclass(frozen=True)
class RemoteObjectValue:
    payload: bytes | bytearray
    evidence: RemoteObjectEvidence


@dataclass(frozen=True)
class RemoteWindowEvidence:
    run_id: str
    source: str
    mode: str
    epoch: int
    sampler_seed: int
    run_receipt: dict[str, Any]
    manifest: RemoteObjectEvidence
    dataset: RemoteObjectEvidence
    cleanup_prepare_receipt: dict[str, str]
    cleanup_commit_receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source": self.source,
            "mode": self.mode,
            "epoch": self.epoch,
            "sampler_seed": self.sampler_seed,
            "run_receipt": dict(self.run_receipt),
            "manifest": self.manifest.to_dict(),
            "dataset": self.dataset.to_dict(),
            "cleanup_prepare_receipt": dict(self.cleanup_prepare_receipt),
            "cleanup_commit_receipt": dict(self.cleanup_commit_receipt),
        }


@dataclass(frozen=True)
class RemoteCheckpointPublication:
    checkpoint_id: str
    checkpoint_target: str
    transaction_manifest_sha256: str
    transaction_commit: dict[str, Any]
    remote_objects: tuple[RemoteObjectEvidence, ...]
    loss_publication: dict[str, Any]
    verification_publication: dict[str, Any]
    pointer_publications: tuple[dict[str, Any], ...]

    def work_artifact(self, bundle: R2MapCheckpointBundle) -> dict[str, Any]:
        manifest = bundle.objects["checkpoint.json"]
        return {
            "label": "candidate-checkpoint",
            "path": f"{self.checkpoint_target}/checkpoint.json",
            "bytes": len(manifest),
            "sha256": content_sha256(manifest),
            "storage_receipt_relative": self.transaction_commit[
                "storage_receipt_relative"
            ],
            "storage_receipt_sha256": self.transaction_commit[
                "storage_receipt_sha256"
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "schema_id": REMOTE_TRAINING_SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_target": self.checkpoint_target,
            "transaction_manifest_sha256": self.transaction_manifest_sha256,
            "transaction_commit": dict(self.transaction_commit),
            "remote_objects": [value.to_dict() for value in self.remote_objects],
            "loss_publication": dict(self.loss_publication),
            "verification_publication": dict(self.verification_publication),
            "pointer_publications": [dict(value) for value in self.pointer_publications],
        }


@dataclass(frozen=True)
class RemoteCheckpointResume:
    bundle: R2MapCheckpointBundle
    loss_content: bytes
    pointer: dict[str, Any]
    pointer_evidence: RemoteObjectEvidence
    loss_evidence: RemoteObjectEvidence
    checkpoint_evidence: tuple[RemoteObjectEvidence, ...]
    verification_receipt: dict[str, Any] | None
    verification_evidence: RemoteObjectEvidence | None


def read_remote_object(
    client: RemoteStorageClient,
    relative: str,
    *,
    maximum_bytes: int,
    as_bytearray: bool = False,
    window_bytes: int = REMOTE_WINDOW_READ_BYTES,
) -> RemoteObjectValue:
    """Read one token-frozen object while preserving every receipt."""
    relative = _remote_relative(relative, "remote object")
    if maximum_bytes < 1 or not 1 <= window_bytes <= REMOTE_WINDOW_READ_BYTES:
        raise ValueError("remote object bounds are invalid")
    opened = client.open_object_with_receipt(relative)
    token = opened.get("object_token")
    if not isinstance(token, Mapping) or token.get("relative") != relative:
        raise R2MapRemoteTrainingError("remote open result does not bind the requested object")
    size = token.get("size")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or size > maximum_bytes
    ):
        raise R2MapRemoteTrainingError("remote object exceeds its in-memory bound")
    open_receipt = _receipt_binding(opened, "remote object open")
    payload = bytearray()
    receipts: list[dict[str, Any]] = []
    expected_offset = 0
    digest = hashlib.sha256()
    for read in client.iter_object_with_receipts(token, window_bytes=window_bytes):
        chunk = read.get("payload")
        if not isinstance(chunk, bytes) or read.get("offset") != expected_offset:
            raise R2MapRemoteTrainingError("remote range sequence is discontinuous")
        if read.get("length") != len(chunk):
            raise R2MapRemoteTrainingError("remote range length differs from payload")
        _receipt_binding(read, "remote object range")
        expected_offset += len(chunk)
        if expected_offset > size:
            raise R2MapRemoteTrainingError("remote range sequence exceeds object size")
        digest.update(chunk)
        payload.extend(chunk)
        receipts.append({key: value for key, value in read.items() if key != "payload"})
    if expected_offset != size or digest.hexdigest() != token.get("sha256"):
        raise R2MapRemoteTrainingError("remote object bytes differ from its frozen token")
    value: bytes | bytearray = payload if as_bytearray else bytes(payload)
    return RemoteObjectValue(
        value,
        RemoteObjectEvidence(
            relative=relative,
            object_token=dict(token),
            open_receipt=open_receipt,
            range_receipts=tuple(receipts),
        ),
    )


def load_remote_compact_index(
    client: RemoteStorageClient,
    relative: str,
    *,
    maximum_bytes: int = MAX_COMPACT_INDEX_BYTES,
) -> tuple[dict[str, Any], RemoteObjectEvidence]:
    value = read_remote_object(client, relative, maximum_bytes=maximum_bytes)
    try:
        decoded = json.loads(value.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R2MapRemoteTrainingError("remote compact index is invalid JSON") from error
    return validate_compact_index_value(decoded), value.evidence


class John2RemoteWindowLoader:
    """Export one John2 source shard and return one cleaned in-memory window."""

    def __init__(
        self,
        client: RemoteStorageClient,
        *,
        exporter_relative: str,
        shard_root_relative: str,
        maximum_window_bytes: int = MAX_IN_MEMORY_STREAM_BYTES,
        timeout_seconds: int = 3_600,
        evidence_sink: Callable[[RemoteWindowEvidence], None] | None = None,
    ):
        self.client = client
        self.exporter_relative = _remote_relative(
            exporter_relative, "remote compact exporter"
        )
        self.shard_root_relative = _remote_relative(
            shard_root_relative, "remote compact shard root"
        )
        if (
            not 1 <= maximum_window_bytes <= MAX_IN_MEMORY_STREAM_BYTES
            or not 1 <= timeout_seconds <= 86_400
        ):
            raise ValueError("remote compact window limits are invalid")
        self.maximum_window_bytes = maximum_window_bytes
        self.timeout_seconds = timeout_seconds
        self.evidence_sink = evidence_sink
        self.evidence: list[RemoteWindowEvidence] = []

    def __call__(
        self,
        source: str,
        mode: str,
        epoch: int,
        sampler_seed: int,
        chunk_index: int,
        game_indices: tuple[int, ...],
    ) -> tuple[dict[str, Any], bytearray]:
        if PurePosixPath(source).name != source or source in {".", ".."}:
            raise R2MapRemoteTrainingError("compact source must be one safe file name")
        if mode not in {"train", "validation"}:
            raise R2MapRemoteTrainingError("remote compact window mode is unsupported")
        if min(epoch, sampler_seed) < 0:
            raise R2MapRemoteTrainingError("remote compact sampler values are negative")
        run_id = f"r2win-{uuid.uuid4().hex}"
        manifest_relative = f"build/run-{run_id}/window.json"
        dataset_relative = f"build/run-{run_id}/window.r2map"
        source_relative = f"{self.shard_root_relative}/{source}"
        run_result: dict[str, Any] | None = None
        outputs: dict[str, Any] | None = None
        cleanup_done = False
        try:
            argv = [
                str(REMOTE_ROOT / self.exporter_relative),
                "export-r2-map-dataset",
                "--shard",
                str(REMOTE_ROOT / source_relative),
                "--manifest",
                str(REMOTE_ROOT / manifest_relative),
                "--stream",
                str(REMOTE_ROOT / dataset_relative),
                "--mode",
                mode,
                "--epoch",
                str(epoch if mode == "train" else 0),
                "--sampler-seed",
                str(sampler_seed if mode == "train" else 0),
            ]
            for game_index in game_indices:
                argv.extend(("--game-index", str(game_index)))
            run_result = self.client.run_remote(
                run_id=run_id,
                cwd_relative=self.shard_root_relative,
                argv=argv,
                output_relative=f"logs/window-exports/{run_id}",
                timeout_seconds=self.timeout_seconds,
            )
            _receipt_binding(run_result, "remote compact exporter run")
            if (
                run_result.get("exit_code") != 0
                or run_result.get("timed_out") is not False
                or run_result.get("temporary_cleaned") is not True
            ):
                raise R2MapRemoteTrainingError("remote compact exporter did not finish cleanly")
            outputs = self.client.open_ephemeral_run_outputs(
                run_id=run_id,
                manifest_relative=manifest_relative,
                dataset_relative=dataset_relative,
            )
            manifest_token = outputs["manifest"]["object_token"]
            dataset_token = outputs["dataset"]["object_token"]
            if dataset_token["size"] > self.maximum_window_bytes:
                raise R2MapRemoteTrainingError("remote compact window exceeds memory ceiling")
            manifest_value = read_remote_object(
                self.client,
                manifest_relative,
                maximum_bytes=2 << 20,
            )
            dataset_value = read_remote_object(
                self.client,
                dataset_relative,
                maximum_bytes=self.maximum_window_bytes,
                as_bytearray=True,
            )
            if (
                manifest_value.evidence.object_token["token_sha256"]
                != manifest_token["token_sha256"]
                or dataset_value.evidence.object_token["token_sha256"]
                != dataset_token["token_sha256"]
            ):
                raise R2MapRemoteTrainingError("ephemeral output changed after it was opened")
            try:
                manifest = json.loads(manifest_value.payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise R2MapRemoteTrainingError(
                    "remote compact manifest is invalid JSON"
                ) from error
            prepared = self.client.prepare_run_cleanup(
                run_id=run_id,
                manifest_object_token=manifest_token,
                dataset_object_token=dataset_token,
            )
            prepare_receipt = _receipt_binding(prepared, "window cleanup prepare")
            committed = self.client.commit_run_cleanup(prepared["cleanup_token"])
            cleanup_done = True
            _receipt_binding(committed, "window cleanup commit")
            if committed.get("cleanup_token_sha256") != prepared["cleanup_token"].get(
                "cleanup_token_sha256"
            ):
                raise R2MapRemoteTrainingError("window cleanup commit differs from prepare")
            evidence = RemoteWindowEvidence(
                run_id=run_id,
                source=source,
                mode=mode,
                epoch=epoch,
                sampler_seed=sampler_seed,
                run_receipt=_without_large_fields(run_result),
                manifest=manifest_value.evidence,
                dataset=dataset_value.evidence,
                cleanup_prepare_receipt=prepare_receipt,
                cleanup_commit_receipt=_without_large_fields(committed),
            )
            if self.evidence_sink is None:
                self.evidence.append(evidence)
            else:
                self.evidence_sink(evidence)
            return manifest, dataset_value.payload  # type: ignore[return-value]
        except BaseException as error:
            cleanup_error = (
                None if cleanup_done else self._recover_failed_window(run_id, outputs)
            )
            if cleanup_error is not None:
                raise R2MapRemoteTrainingError(
                    f"remote compact window failed and cleanup also failed: {cleanup_error}"
                ) from error
            raise

    def _recover_failed_window(
        self, run_id: str, outputs: Mapping[str, Any] | None
    ) -> BaseException | None:
        first_error: BaseException | None = None
        if outputs is not None:
            try:
                prepared = self.client.prepare_run_cleanup(
                    run_id=run_id,
                    manifest_object_token=outputs["manifest"]["object_token"],
                    dataset_object_token=outputs["dataset"]["object_token"],
                )
                self.client.commit_run_cleanup(prepared["cleanup_token"])
                return None
            except BaseException as cleanup_error:  # pragma: no cover - live fault path
                first_error = cleanup_error
        try:
            prepared = self.client.prepare_failed_run_cleanup(run_id=run_id)
            self.client.commit_failed_run_cleanup(prepared["cleanup_token"])
        except BaseException as cleanup_error:  # pragma: no cover - live transport fault
            if isinstance(cleanup_error, RemoteOperationError) and str(cleanup_error) == (
                "failed-run cleanup found no exact run trees"
            ):
                return None
            return first_error or cleanup_error
        return None


class John2RemoteCheckpointStore:
    """Atomically publish and resume in-memory MLX checkpoints on John2."""

    def __init__(self, client: RemoteStorageClient, *, run_id: str):
        self.client = client
        self.run_id = _remote_identifier(run_id, "training run")
        self._mutable_sha256: dict[str, str] = {}

    @property
    def loss_relative(self) -> str:
        return f"runs/{self.run_id}/losses/loss-stream.jsonl"

    def publish_immutable_json(
        self, relative: str, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        relative = _remote_relative(relative, "immutable training JSON")
        payload = _json_line(value)
        publication = self.client.put_bytes(
            relative,
            payload,
            expected_current="absent",
        )
        _receipt_binding(publication, "immutable training JSON publication")
        if (
            publication.get("mode") != "0o400"
            or publication.get("sha256") != content_sha256(payload)
        ):
            raise R2MapRemoteTrainingError("immutable training JSON publication differs")
        return dict(publication)

    def publish_loss_stream(self, content: bytes) -> dict[str, Any]:
        if not isinstance(content, bytes):
            raise TypeError("loss stream must be immutable bytes")
        relative = self.loss_relative
        result = self.client.put_bytes(
            relative,
            content,
            expected_current=self._mutable_sha256.get(relative, "absent"),
            mutable=True,
        )
        _receipt_binding(result, "loss stream publication")
        if result.get("mode") != "0o600" or result.get("sha256") != content_sha256(content):
            raise R2MapRemoteTrainingError("loss stream publication identity differs")
        self._mutable_sha256[relative] = result["sha256"]
        return dict(result)

    def publish_checkpoint(
        self,
        bundle: R2MapCheckpointBundle,
        *,
        loss_content: bytes,
        verification_receipt: Mapping[str, Any],
        pointer_names: Sequence[str] = ("latest_complete", "last_verified"),
    ) -> RemoteCheckpointPublication:
        _, state, _ = verify_r2_map_checkpoint_bundle(bundle, loss_stream=loss_content)
        validate_verification_receipt_value(
            verification_receipt,
            checkpoint_id=bundle.checkpoint_id,
            checkpoint_manifest_blake3=bundle.manifest_blake3,
            expected_dataset_contract_blake3=_canonical_blake3(state.dataset_contract),
        )
        if bundle.total_bytes > MAX_CHECKPOINT_BUNDLE_BYTES:
            raise R2MapRemoteTrainingError("checkpoint exceeds the in-memory bundle bound")
        _remote_identifier(bundle.checkpoint_id, "checkpoint")
        loss_publication = self.publish_loss_stream(loss_content)
        target = f"runs/{self.run_id}/checkpoints/{bundle.checkpoint_id}"
        objects = [
            TransactionObject(name, len(payload), content_sha256(payload))
            for name, payload in sorted(bundle.objects.items())
        ]
        transaction_id = f"ckpt-{content_sha256(bundle.objects['checkpoint.json'])[:48]}"
        manifest = build_transaction_manifest(transaction_id, target, objects)
        begun = False
        committed = False
        try:
            begin = self.client.begin_transaction(manifest)
            _receipt_binding(begin, "checkpoint transaction begin")
            begun = True
            for descriptor in objects:
                result = self.client.put_transaction_object(
                    transaction_id,
                    descriptor,
                    [bundle.objects[descriptor.relative]],
                )
                _receipt_binding(result, "checkpoint transaction object")
            commit = self.client.commit_transaction(
                transaction_id, manifest["manifest_sha256"]
            )
            _receipt_binding(commit, "checkpoint transaction commit")
            if (
                commit.get("committed") is not True
                or commit.get("target_relative") != target
                or commit.get("manifest_sha256") != manifest["manifest_sha256"]
            ):
                raise R2MapRemoteTrainingError("checkpoint commit identity differs")
            committed = True
        except BaseException:
            if begun and not committed:
                with suppress(RemoteOperationError, RemoteProtocolError):
                    self.client.abort_transaction(
                        transaction_id, manifest["manifest_sha256"]
                    )
            raise

        remote_objects = self._verify_checkpoint_target(target, bundle, manifest)
        verification_payload = _json_line(dict(verification_receipt))
        verification_relative = f"runs/{self.run_id}/verifications/{bundle.checkpoint_id}.json"
        verification_publication = self.client.put_bytes(
            verification_relative,
            verification_payload,
            expected_current="absent",
        )
        _receipt_binding(verification_publication, "verification receipt publication")
        verification_value = read_remote_object(
            self.client,
            verification_relative,
            maximum_bytes=MAX_VERIFICATION_RECEIPT_BYTES,
        )
        if verification_value.payload != verification_payload:
            raise R2MapRemoteTrainingError("remote verification receipt bytes differ")

        pointer_publications = []
        for name in pointer_names:
            metadata = (
                {"verification_id": verification_receipt["verification_id"]}
                if name == "last_verified"
                else None
            )
            pointer_publications.append(
                self.publish_pointer(name, bundle, metadata=metadata)
            )
        return RemoteCheckpointPublication(
            checkpoint_id=bundle.checkpoint_id,
            checkpoint_target=target,
            transaction_manifest_sha256=manifest["manifest_sha256"],
            transaction_commit=dict(commit),
            remote_objects=remote_objects,
            loss_publication=loss_publication,
            verification_publication=dict(verification_publication),
            pointer_publications=tuple(pointer_publications),
        )

    def publish_pointer(
        self,
        name: str,
        bundle: R2MapCheckpointBundle,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        pointer = build_r2_map_checkpoint_pointer_document(
            name, bundle, metadata=metadata
        )
        payload = _json_line(pointer, pretty=True)
        relative = f"runs/{self.run_id}/{name}.json"
        publication = self.client.put_bytes(
            relative,
            payload,
            expected_current=self._mutable_sha256.get(relative, "absent"),
            mutable=True,
        )
        _receipt_binding(publication, f"{name} pointer publication")
        if publication.get("mode") != "0o600":
            raise R2MapRemoteTrainingError("checkpoint pointer is not mutable mode 0600")
        self._mutable_sha256[relative] = publication["sha256"]
        return dict(publication)

    def load_checkpoint(self, pointer_name: str) -> RemoteCheckpointResume:
        pointer_relative = f"runs/{self.run_id}/{pointer_name}.json"
        pointer_value = read_remote_object(
            self.client, pointer_relative, maximum_bytes=MAX_VERIFICATION_RECEIPT_BYTES
        )
        try:
            pointer = json.loads(pointer_value.payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise R2MapRemoteTrainingError("remote checkpoint pointer is invalid JSON") from error
        checkpoint_id = _remote_identifier(pointer.get("checkpoint"), "checkpoint pointer")
        target = f"runs/{self.run_id}/checkpoints/{checkpoint_id}"
        names = (
            "checkpoint.json",
            "model.safetensors",
            "optimizer.safetensors",
            "state.json",
            "fixed-prediction-panel.safetensors",
        )
        objects: dict[str, bytes] = {}
        evidence = []
        total = 0
        for name in names:
            value = read_remote_object(
                self.client,
                f"{target}/{name}",
                maximum_bytes=MAX_CHECKPOINT_OBJECT_BYTES,
            )
            objects[name] = bytes(value.payload)
            evidence.append(value.evidence)
            total += len(value.payload)
            if total > MAX_CHECKPOINT_BUNDLE_BYTES:
                raise R2MapRemoteTrainingError("remote checkpoint bundle exceeds memory bound")
        try:
            manifest = json.loads(objects["checkpoint.json"])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise R2MapRemoteTrainingError("remote checkpoint manifest is invalid JSON") from error
        bundle = R2MapCheckpointBundle(checkpoint_id, manifest, objects)
        validate_r2_map_checkpoint_pointer_document(
            pointer, name=pointer_name, bundle=bundle
        )
        loss = read_remote_object(
            self.client,
            self.loss_relative,
            maximum_bytes=MAX_CHECKPOINT_OBJECT_BYTES,
        )
        _, state, _ = verify_r2_map_checkpoint_bundle(
            bundle, loss_stream=bytes(loss.payload)
        )
        verification_receipt = None
        verification_evidence = None
        if pointer_name in {"last_verified", "best_validation"}:
            verification = read_remote_object(
                self.client,
                f"runs/{self.run_id}/verifications/{checkpoint_id}.json",
                maximum_bytes=MAX_VERIFICATION_RECEIPT_BYTES,
            )
            try:
                verification_receipt = json.loads(verification.payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise R2MapRemoteTrainingError(
                    "remote checkpoint verification receipt is invalid JSON"
                ) from error
            verification_receipt = validate_verification_receipt_value(
                verification_receipt,
                checkpoint_id=checkpoint_id,
                checkpoint_manifest_blake3=bundle.manifest_blake3,
                expected_dataset_contract_blake3=_canonical_blake3(state.dataset_contract),
            )
            verification_evidence = verification.evidence
            if pointer_name == "last_verified" and pointer.get("metadata") != {
                "verification_id": verification_receipt["verification_id"]
            }:
                raise R2MapRemoteTrainingError(
                    "last_verified pointer does not bind its verification receipt"
                )
        self._mutable_sha256[pointer_relative] = pointer_value.evidence.object_token["sha256"]
        self._mutable_sha256[self.loss_relative] = loss.evidence.object_token["sha256"]
        return RemoteCheckpointResume(
            bundle=bundle,
            loss_content=bytes(loss.payload),
            pointer=pointer,
            pointer_evidence=pointer_value.evidence,
            loss_evidence=loss.evidence,
            checkpoint_evidence=tuple(evidence),
            verification_receipt=verification_receipt,
            verification_evidence=verification_evidence,
        )

    def _verify_checkpoint_target(
        self,
        target: str,
        bundle: R2MapCheckpointBundle,
        transaction_manifest: Mapping[str, Any],
    ) -> tuple[RemoteObjectEvidence, ...]:
        evidence = []
        for name, expected in sorted(bundle.objects.items()):
            observed = read_remote_object(
                self.client,
                f"{target}/{name}",
                maximum_bytes=MAX_CHECKPOINT_OBJECT_BYTES,
            )
            if observed.payload != expected:
                raise R2MapRemoteTrainingError(f"remote checkpoint object differs: {name}")
            evidence.append(observed.evidence)
        provenance = read_remote_object(
            self.client,
            f"{target}/.r2-map-transaction.json",
            maximum_bytes=2 << 20,
        )
        try:
            decoded = json.loads(provenance.payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise R2MapRemoteTrainingError("checkpoint provenance is invalid JSON") from error
        if decoded != dict(transaction_manifest) or canonical_json(decoded) != provenance.payload:
            raise R2MapRemoteTrainingError("checkpoint provenance manifest differs")
        evidence.append(provenance.evidence)
        return tuple(evidence)


def _receipt_binding(value: Mapping[str, Any], label: str) -> dict[str, str]:
    relative = value.get("storage_receipt_relative")
    digest = value.get("storage_receipt_sha256")
    if (
        not isinstance(relative, str)
        or not relative.startswith("control/receipts/req-")
        or not relative.endswith(".json")
        or not _sha256(digest)
    ):
        raise R2MapRemoteTrainingError(f"{label} lacks persisted John2 receipt evidence")
    return {
        "storage_receipt_relative": relative,
        "storage_receipt_sha256": digest,
    }


def _without_large_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"cleanup_token", "payload"}}


def _remote_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{label} must be a canonical relative path")
    return value


def _remote_identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or value[0] not in "abcdefghijklmnopqrstuvwxyz0123456789"
        or value[-1] not in "abcdefghijklmnopqrstuvwxyz0123456789"
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in value)
    ):
        raise R2MapRemoteTrainingError(f"{label} is not a safe remote identifier")
    return value


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_line(value: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        return canonical_json(value) + b"\n"
    except (TypeError, ValueError) as error:
        raise R2MapRemoteTrainingError("remote training JSON is not finite/canonical") from error


def finite_validation_metric(receipt: Mapping[str, Any]) -> float:
    """Extract a finite primary loss for deterministic best-checkpoint selection."""
    metric = receipt.get("validation", {}).get("primary_score_to_go_loss")
    if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
        raise R2MapRemoteTrainingError("checkpoint publication lacks finite primary validation")
    return float(metric)
