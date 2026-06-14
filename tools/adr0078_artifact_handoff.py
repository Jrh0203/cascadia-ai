"""Dataset ownership and atomic handoff rules for ADR 0078."""

from __future__ import annotations

import filecmp
import json
import os
import shutil
from pathlib import Path

import adr0078_cluster_runtime as rt

IMMUTABLE_MANIFEST_FIELDS = (
    "schema_version",
    "dataset_id",
    "feature_schema",
    "target_schema",
    "record_size",
    "action_position_record_size",
    "target_dim",
    "maximum_candidates",
    "maximum_samples",
    "game",
    "split",
    "teacher",
    "first_game_index",
    "requested_games",
)


def assert_no_unregistered_local_validation_collector() -> None:
    result = rt.run(
        [str(rt.PGREP), "-f", rt.VALIDATION_SPEC.process_pattern],
        check=False,
        quiet=True,
    )
    if result.returncode == 0:
        raise RuntimeError(
            "an unregistered validation collector is running on john1; "
            "ADR 0078 assigns validation exclusively to john2"
        )
    if result.returncode != 1:
        raise RuntimeError(f"could not verify john1 validation ownership: exit {result.returncode}")


def install_validated_dataset(
    incoming: Path,
    destination: Path,
    invalidated_root: Path,
) -> Path | None:
    if not destination.exists():
        os.replace(incoming, destination)
        return None

    destination_manifest = destination / "dataset.json"
    incoming_manifest = incoming / "dataset.json"
    if rt.sha256_file(destination_manifest) == rt.sha256_file(incoming_manifest):
        shutil.rmtree(incoming)
        return None

    existing = json.loads(destination_manifest.read_text())
    complete = json.loads(incoming_manifest.read_text())
    _require_strict_prefix(destination, existing, incoming, complete)

    manifest_sha256 = rt.sha256_file(destination_manifest)
    archive = invalidated_root / (f"adr-0078-unregistered-john1-validation-{manifest_sha256[:12]}")
    if archive.exists():
        raise RuntimeError(f"collision archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    os.replace(destination, archive)
    rt.atomic_json(
        archive / "invalidation.json",
        {
            "archived_at": rt.timestamp(),
            "reason": (
                "unregistered john1 validation collector duplicated a strict "
                "prefix of the john2-owned complete corpus"
            ),
            "destination_manifest_sha256": manifest_sha256,
            "registered_manifest_sha256": rt.sha256_file(incoming_manifest),
            "overlap_shards": len(existing["shards"]),
            "registered_shards": len(complete["shards"]),
        },
    )
    os.replace(incoming, destination)
    return archive


def _require_strict_prefix(
    existing_root: Path,
    existing: dict,
    incoming_root: Path,
    incoming: dict,
) -> None:
    for field in IMMUTABLE_MANIFEST_FIELDS:
        if existing.get(field) != incoming.get(field):
            raise ValueError(f"existing john1 validation dataset changed {field}")

    existing_games = int(existing.get("completed_games", -1))
    incoming_games = int(incoming.get("completed_games", -1))
    if not 0 <= existing_games < incoming_games:
        raise ValueError("existing john1 validation dataset is not a strict prefix")

    existing_shards = existing.get("shards")
    incoming_shards = incoming.get("shards")
    if not isinstance(existing_shards, list) or not isinstance(incoming_shards, list):
        raise ValueError("validation dataset is missing shard metadata")
    if existing_shards != incoming_shards[: len(existing_shards)]:
        raise ValueError("existing john1 validation shard metadata is not a strict prefix")

    for shard in existing_shards:
        name = shard.get("file")
        if not isinstance(name, str):
            raise ValueError("validation shard has an invalid filename")
        left = existing_root / name
        right = incoming_root / name
        if not left.is_file() or not right.is_file():
            raise ValueError(f"validation prefix shard is missing: {name}")
        if not filecmp.cmp(left, right, shallow=False):
            raise ValueError(f"validation prefix shard differs: {name}")
