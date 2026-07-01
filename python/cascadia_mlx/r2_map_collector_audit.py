"""Lightweight completion audit for an already semantically validated R2-MAP collector.

The Rust collector performs the expensive replay/example validation before it
returns successfully.  This module deliberately does not replay records.  It
binds the collector's successful JSON payload to the durable manifest, verifies
the complete shard lease and every recorded byte count/BLAKE3 digest, and emits
an atomic completion receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import blake3


class CollectorAuditError(RuntimeError):
    """A collector completion artifact failed closed."""


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CollectorAuditError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise CollectorAuditError(f"JSON document is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _blake3(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _write_copy_manifest(dataset: Path, path: Path) -> None:
    files = sorted(
        candidate
        for candidate in dataset.iterdir()
        if candidate.is_file() and candidate.resolve() != path.resolve()
    )
    lines = [f"{_sha256(candidate)}  ./{candidate.name}" for candidate in files]
    _atomic_text(path, "\n".join(lines) + "\n")


def _payload_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload.get("manifest", payload)
    if not isinstance(candidate, dict):
        raise CollectorAuditError("collector validation payload has no manifest object")
    return candidate


def audit(
    *,
    dataset: Path,
    validation_payload: Path,
    validation_manifest: Path,
    receipt_path: Path,
    semantic_validation_proof: str,
    copy_manifest: Path | None = None,
) -> dict[str, Any]:
    dataset = dataset.resolve(strict=True)
    manifest_path = dataset / "dataset.json"
    manifest = _object(manifest_path)
    payload = _object(validation_payload)
    if _payload_manifest(payload) != manifest:
        raise CollectorAuditError("collector validation payload differs from dataset manifest")

    lease = manifest.get("lease")
    shards = manifest.get("shards")
    collection_kind = manifest.get("collection_kind")
    primary_per_game = {"bootstrap": 80, "iterative-training": 20}.get(collection_kind)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("trajectory_schema_version") != 2
        or not isinstance(manifest.get("dataset_id"), str)
        or not manifest["dataset_id"]
        or not isinstance(lease, dict)
        or not isinstance(shards, list)
        or not shards
        or primary_per_game is None
    ):
        raise CollectorAuditError("collector manifest schema or identity differs")
    requested = manifest.get("requested_games")
    completed = manifest.get("completed_games")
    shard_games = manifest.get("shard_games")
    if (
        not isinstance(requested, int)
        or requested <= 0
        or completed != requested
        or lease.get("game_count") != requested
        or not isinstance(shard_games, int)
        or shard_games <= 0
    ):
        raise CollectorAuditError("collector lease is not complete")
    expected_purpose = "bootstrap" if collection_kind == "bootstrap" else "generation"
    if lease.get("purpose") != expected_purpose:
        raise CollectorAuditError("collector lease purpose differs")

    cursor = lease.get("first_game_index")
    if not isinstance(cursor, int) or cursor < 0:
        raise CollectorAuditError("collector lease start is invalid")
    total_games = total_examples = total_bytes = 0
    shard_identities: list[dict[str, Any]] = []
    for ordinal, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise CollectorAuditError("collector shard manifest is not an object")
        file_name = f"shard-{ordinal:05d}.r2sh"
        game_count = shard.get("game_count")
        if (
            shard.get("file") != file_name
            or shard.get("first_game_index") != cursor
            or not isinstance(game_count, int)
            or game_count <= 0
            or game_count > shard_games
            or shard.get("primary_example_count") != game_count * primary_per_game
        ):
            raise CollectorAuditError(f"collector shard sequence differs: {file_name}")
        path = (dataset / file_name).resolve(strict=True)
        try:
            path.relative_to(dataset)
        except ValueError as error:
            raise CollectorAuditError("collector shard escapes dataset root") from error
        byte_count = path.stat().st_size
        digest = _blake3(path)
        if byte_count != shard.get("byte_count") or digest != shard.get("blake3"):
            raise CollectorAuditError(f"collector shard hash differs: {path}")
        shard_identities.append(
            {"file": file_name, "byte_count": byte_count, "blake3": digest}
        )
        cursor += game_count
        total_games += game_count
        total_examples += game_count * primary_per_game
        total_bytes += byte_count

    if (
        total_games != requested
        or total_examples != manifest.get("primary_example_count")
        or cursor != lease["first_game_index"] + lease["game_count"]
    ):
        raise CollectorAuditError("collector shard aggregates differ from manifest")

    expected_files = {
        ".collector.lock",
        "dataset.json",
        "validation.json",
        "completion-audit.json",
        "copied-files.sha256",
        *(shard["file"] for shard in shards),
    }
    actual_files = {path.name for path in dataset.iterdir() if path.is_file()}
    unexpected = actual_files - expected_files
    if unexpected or any(path.is_dir() for path in dataset.iterdir()):
        raise CollectorAuditError(f"collector directory has unexpected artifacts: {sorted(unexpected)}")

    receipt: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.collector-completion-audit.v1",
        "schema_version": 1,
        "result": "pass",
        "semantic_validation_proof": semantic_validation_proof,
        "dataset_id": manifest["dataset_id"],
        "collection_kind": collection_kind,
        "host_id": lease.get("host_id"),
        "first_game_index": lease["first_game_index"],
        "game_count": requested,
        "primary_example_count": total_examples,
        "shard_count": len(shards),
        "shard_bytes": total_bytes,
        "dataset_json_sha256": _sha256(manifest_path),
        "collector_validation_payload_sha256": _sha256(validation_payload),
        "shards": shard_identities,
        "audited_unix_ms": time.time_ns() // 1_000_000,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(validation_manifest, manifest)
    _atomic_json(receipt_path, receipt)
    if copy_manifest is not None:
        _write_copy_manifest(dataset, copy_manifest)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-payload", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--copy-manifest", type=Path)
    parser.add_argument(
        "--semantic-validation-proof",
        choices=("collector-zero-exit", "observed-validator-transition"),
        required=True,
    )
    args = parser.parse_args()
    receipt = audit(
        dataset=args.dataset,
        validation_payload=args.validation_payload,
        validation_manifest=args.validation_manifest,
        receipt_path=args.receipt,
        semantic_validation_proof=args.semantic_validation_proof,
        copy_manifest=args.copy_manifest,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
