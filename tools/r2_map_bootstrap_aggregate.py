#!/usr/bin/env python3
"""Validate and flatten a complete multi-worker R2-MAP bootstrap corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import blake3


class AggregateError(RuntimeError):
    """The bootstrap aggregate failed closed."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AggregateError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise AggregateError(f"JSON document is not an object: {path}")
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


def _verify_completion_audit(
    root: Path, manifest: dict[str, Any], validation_path: Path
) -> dict[str, Any]:
    path = root / "completion-audit.json"
    receipt = _json(path)
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    actual_receipt = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lease = manifest["lease"]
    expected_shards = [
        {
            "file": shard["file"],
            "byte_count": shard["byte_count"],
            "blake3": shard["blake3"],
        }
        for shard in manifest["shards"]
    ]
    if (
        receipt.get("schema_id")
        != "cascadia.r2-map.collector-completion-audit.v1"
        or receipt.get("schema_version") != 1
        or receipt.get("result") != "pass"
        or receipt.get("semantic_validation_proof")
        not in {"collector-zero-exit", "observed-validator-transition"}
        or receipt.get("dataset_id") != manifest["dataset_id"]
        or receipt.get("collection_kind") != manifest["collection_kind"]
        or receipt.get("host_id") != lease["host_id"]
        or receipt.get("first_game_index") != lease["first_game_index"]
        or receipt.get("game_count") != manifest["completed_games"]
        or receipt.get("primary_example_count") != manifest["primary_example_count"]
        or receipt.get("shard_count") != len(manifest["shards"])
        or receipt.get("shard_bytes")
        != sum(shard["byte_count"] for shard in manifest["shards"])
        or receipt.get("dataset_json_sha256") != _sha256(root / "dataset.json")
        or receipt.get("shards") != expected_shards
        or claimed != actual_receipt
    ):
        raise AggregateError(f"collector completion audit differs: {path}")
    if receipt["semantic_validation_proof"] == "collector-zero-exit":
        payload_hashes = {
            _sha256(root / "dataset.json"),
            _sha256(validation_path),
        }
        if receipt.get("collector_validation_payload_sha256") not in payload_hashes:
            # Iteration collector stdout is an envelope around the manifest and is
            # intentionally removed after the receipt binds it.
            if receipt.get("collection_kind") == "bootstrap":
                raise AggregateError("bootstrap collector payload hash is not reproducible")
    return receipt


def _protocol_bytes(value: str) -> list[int]:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise AggregateError("packet protocol identity is not a lowercase 32-byte digest")
    return list(bytes.fromhex(value))


def _verify_copy_manifest(root: Path) -> int:
    path = root / "copied-files.sha256"
    if not path.is_file():
        return 0
    entries = 0
    listed: set[str] = set()
    for line in path.read_text().splitlines():
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise AggregateError(f"invalid copied-file checksum line: {path}") from error
        relative = relative.lstrip("* ")
        normalized = Path(relative).as_posix().removeprefix("./")
        if not normalized or normalized in listed:
            raise AggregateError("copied-file checksum repeats or omits a path")
        listed.add(normalized)
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root.resolve(strict=True))
        except ValueError as error:
            raise AggregateError("copied-file checksum escapes worker root") from error
        if _sha256(candidate) != expected:
            raise AggregateError(f"copied-file SHA-256 differs: {candidate}")
        entries += 1
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "copied-files.sha256"
    }
    if listed != actual:
        raise AggregateError("copied-file checksum coverage differs from worker files")
    return entries


def _expected_workers(packet: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    hosts = packet.get("hosts")
    if not isinstance(hosts, dict) or set(hosts) != {"john1", "john2", "john3"}:
        raise AggregateError("packet must name exactly john1, john2, and john3")
    workers: list[tuple[str, dict[str, Any]]] = []
    cursor = 0
    for host in ("john1", "john2", "john3"):
        values = hosts[host].get("workers")
        if not isinstance(values, list) or not values:
            raise AggregateError(f"packet has no workers for {host}")
        for worker in values:
            required = {"worker", "first_game_index", "games", "last_game_index"}
            if not isinstance(worker, dict) or set(worker) != required:
                raise AggregateError("packet worker schema differs")
            if (
                worker["first_game_index"] != cursor
                or worker["games"] <= 0
                or worker["last_game_index"] != cursor + worker["games"] - 1
            ):
                raise AggregateError("packet worker ranges are not exact and contiguous")
            workers.append((host, worker))
            cursor += worker["games"]
    if cursor != packet.get("games_total"):
        raise AggregateError("packet aggregate game count differs")
    return workers


def aggregate(
    *,
    packet_path: Path,
    host_roots: dict[str, Path],
    output_path: Path,
    flat_shard_root: Path,
    require_worker_validation: bool,
) -> dict[str, Any]:
    packet = _json(packet_path)
    workers = _expected_workers(packet)
    if set(host_roots) != {"john1", "john2", "john3"}:
        raise AggregateError("host roots must name exactly john1, john2, and john3")
    immutable = packet.get("immutable_identity", {})
    expected_protocols = {
        "collector_hash": _protocol_bytes(immutable.get("collector_binary_blake3", "")),
        "source_hash": _protocol_bytes(immutable.get("source_sha256", "")),
        "serving_protocol_hash": _protocol_bytes(immutable.get("serving_protocol_blake3", "")),
    }
    if output_path.exists() or flat_shard_root.exists():
        raise AggregateError("aggregate output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_shard_root.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{flat_shard_root.name}.", dir=flat_shard_root.parent)
    )
    datasets: list[dict[str, Any]] = []
    flattened: list[dict[str, Any]] = []
    total_examples = total_games = copied_entries = validation_payloads = completion_audits = 0
    dataset_ids: set[str] = set()
    try:
        for host, spec in workers:
            worker = spec["worker"]
            root = host_roots[host] / f"worker-{worker}"
            manifest_path = root / "dataset.json"
            manifest = _json(manifest_path)
            lease = manifest.get("lease", {})
            games = spec["games"]
            if (
                manifest.get("schema_version") != 1
                or manifest.get("trajectory_schema_version") != 2
                or manifest.get("collection_kind") != "bootstrap"
                or manifest.get("policy") != {"kind": "bootstrap-greedy"}
                or manifest.get("protocols") != expected_protocols
                or lease.get("campaign_id") != packet.get("campaign_id")
                or lease.get("iteration") != packet.get("iteration")
                or lease.get("purpose") != "bootstrap"
                or lease.get("host_id") != host
                or lease.get("first_game_index") != spec["first_game_index"]
                or lease.get("game_count") != games
                or manifest.get("requested_games") != games
                or manifest.get("completed_games") != games
                or manifest.get("primary_example_count") != games * 80
            ):
                raise AggregateError(f"worker manifest contract differs: {manifest_path}")
            game = manifest.get("game", {})
            if (
                game.get("player_count") != 4
                or game.get("mode") != "Standard"
                or game.get("habitat_bonuses") is not False
                or game.get("scoring_cards")
                != {animal: "A" for animal in ("bear", "elk", "salmon", "hawk", "fox")}
            ):
                raise AggregateError("worker game configuration differs")
            cursor = spec["first_game_index"]
            worker_examples = 0
            shards = manifest.get("shards")
            if not isinstance(shards, list) or not shards:
                raise AggregateError("completed worker has no replay shards")
            for shard_ordinal, shard in enumerate(shards):
                source = root / shard["file"]
                try:
                    source.resolve(strict=True).relative_to(root.resolve(strict=True))
                except (OSError, ValueError) as error:
                    raise AggregateError("worker shard escapes its dataset root") from error
                if (
                    shard.get("first_game_index") != cursor
                    or shard.get("game_count", 0) <= 0
                    or shard.get("primary_example_count") != shard["game_count"] * 80
                    or source.stat().st_size != shard.get("byte_count")
                    or _blake3(source) != shard.get("blake3")
                ):
                    raise AggregateError(f"worker shard contract differs: {source}")
                flat_name = f"{host}-w{worker:02d}-s{shard_ordinal:05d}.r2sh"
                assert staging is not None
                destination = staging / flat_name
                os.link(source, destination)
                flattened.append(
                    {
                        "file_name": flat_name,
                        "source": str(source),
                        "first_game_index": cursor,
                        "game_count": shard["game_count"],
                        "bytes": shard["byte_count"],
                        "blake3": shard["blake3"],
                    }
                )
                cursor += shard["game_count"]
                worker_examples += shard["primary_example_count"]
            if cursor != spec["last_game_index"] + 1 or worker_examples != games * 80:
                raise AggregateError("worker shard coverage differs from lease")
            copied_entries += _verify_copy_manifest(root)
            validation_path = root / "validation.json"
            if validation_path.is_file():
                if _json(validation_path) != manifest:
                    raise AggregateError("worker validation payload differs from manifest")
                validation_payloads += 1
            elif require_worker_validation:
                raise AggregateError(f"worker validation payload is absent: {root}")
            if require_worker_validation:
                _verify_completion_audit(root, manifest, validation_path)
                completion_audits += 1
                expected_files = {
                    ".collector.lock",
                    "dataset.json",
                    "validation.json",
                    "completion-audit.json",
                    "copied-files.sha256",
                    *(shard["file"] for shard in shards),
                }
                actual_files = {path.name for path in root.iterdir() if path.is_file()}
                if actual_files != expected_files or any(path.is_dir() for path in root.iterdir()):
                    raise AggregateError("completed worker contains unexpected artifacts")
            dataset_id = manifest.get("dataset_id")
            if not isinstance(dataset_id, str) or not dataset_id or dataset_id in dataset_ids:
                raise AggregateError("worker dataset identity is absent or repeated")
            dataset_ids.add(dataset_id)
            datasets.append(
                {
                    "host": host,
                    "worker": worker,
                    "dataset_id": dataset_id,
                    "first_game_index": spec["first_game_index"],
                    "game_count": games,
                    "primary_example_count": games * 80,
                    "dataset_json_sha256": _sha256(manifest_path),
                    "shard_count": len(shards),
                }
            )
            total_games += games
            total_examples += games * 80
        if total_games != packet["games_total"] or total_examples != total_games * 80:
            raise AggregateError("aggregate totals differ")
        assert staging is not None
        os.replace(staging, flat_shard_root)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    receipt: dict[str, Any] = {
        "schema_id": "cascadia.r2-map.bootstrap-aggregate.v1",
        "schema_version": 1,
        "campaign_id": packet["campaign_id"],
        "run_id": packet["run_id"],
        "packet_sha256": _sha256(packet_path),
        "validated_unix_ms": time.time_ns() // 1_000_000,
        "games": total_games,
        "global_game_indices": f"0-{total_games - 1}",
        "primary_example_count": total_examples,
        "worker_datasets": len(datasets),
        "replay_shards": len(flattened),
        "copied_file_checksum_entries": copied_entries,
        "validation_payloads": validation_payloads,
        "completion_audits": completion_audits,
        "require_worker_validation": require_worker_validation,
        "flat_shard_root": str(flat_shard_root),
        "datasets": datasets,
        "shards": flattened,
        "result": "pass",
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, output_path)
    return receipt


def _host_root(value: str) -> tuple[str, Path]:
    try:
        host, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("host root must be HOST=PATH") from error
    if host not in {"john1", "john2", "john3"} or not raw_path:
        raise argparse.ArgumentTypeError("host root is invalid")
    return host, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--host-root", action="append", type=_host_root, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flat-shard-root", type=Path, required=True)
    parser.add_argument("--require-worker-validation", action="store_true")
    arguments = parser.parse_args()
    roots = dict(arguments.host_root)
    if len(roots) != len(arguments.host_root):
        parser.error("host roots repeat")
    receipt = aggregate(
        packet_path=arguments.packet,
        host_roots=roots,
        output_path=arguments.output,
        flat_shard_root=arguments.flat_shard_root,
        require_worker_validation=arguments.require_worker_validation,
    )
    print(json.dumps({key: receipt[key] for key in ("games", "primary_example_count", "worker_datasets", "replay_shards", "receipt_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
