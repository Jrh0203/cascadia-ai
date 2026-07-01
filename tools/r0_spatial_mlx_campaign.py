#!/usr/bin/env python3
"""Freeze, authorize, preflight, queue, and collect the R0 MLX tournament."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
from cascadia_mlx.dataset import Dataset
from cascadia_mlx.r0_spatial_mlx_cache import (
    ARM_TOKEN_CAPACITY,
    CORPUS_LOCK_CONTRACT,
    CORPUS_LOCK_SCHEMA_VERSION,
    EXPERIMENT_ID,
    R0SpatialMlxCache,
)
from cascadia_mlx.r0_spatial_mlx_model import R0SpatialIsoValueModel, parameter_count
from cascadia_mlx.r0_spatial_mlx_tournament import (
    ADR_ID,
    AUTHORIZED_ARMS,
    PROTOCOL_ID,
    R0SpatialMlxTournamentConfig,
    R0SpatialMlxTournamentProtocol,
    run_tournament,
)
from cascadia_mlx.run_manifest import source_provenance
from cluster_research_queue import add_task, load_queue, locked_queue
from rust_experiment_bundle import BundleError, file_blake3, validate_bundle

HOSTS = ("john1", "john2", "john3", "john4")
REMOTE_ROOTS = {
    "john1": Path("/Users/johnherrick/cascadia"),
    "john2": Path("/Users/john2/cascadia-bench"),
    "john3": Path("/Users/john3/cascadia-bench"),
    "john4": Path("/Users/john4/cascadia-bench"),
}
PRIMARY_ARMS = (
    ("exact-entity-control", "john1"),
    ("hex-radius-6-127", "john2"),
    ("hex-radius-5-91", "john3"),
    ("hex-radius-4-61", "john4"),
)
DIAGNOSTIC_ARM = "historical-square-21x21-441"
ARM_ORDER = tuple(ARM_TOKEN_CAPACITY)
TASK_PREFIX = "r0mlx"
CORPUS_DIGEST_PREFIX = b"R0MLXCORPUS1\0"
STRATEGY_ID = "pattern-aware-v1-k8-h6-b8-m4"
FEATURE_SCHEMA = "compact-entity-v2"
TARGET_SCHEMA = "base-score-components-v1"
DEFAULT_QUEUE = Path("artifacts/cluster/research-queue-v1.json")
DEFAULT_EXPERIMENT_ROOT = Path("artifacts/experiments") / EXPERIMENT_ID
DEFAULT_DATASET_ROOT = Path("artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen")
DEFAULT_CORPUS_LOCK = DEFAULT_EXPERIMENT_ROOT / "control/corpus-lock.json"
DEFAULT_AUTHORIZATION = DEFAULT_EXPERIMENT_ROOT / "control/authorization.json"
DEFAULT_CACHE_ROOT = DEFAULT_EXPERIMENT_ROOT / "caches"
DEFAULT_RUN_ROOT = DEFAULT_EXPERIMENT_ROOT / "runs"
DEFAULT_REPORT_ROOT = DEFAULT_EXPERIMENT_ROOT / "reports"
EXPORTER_BINARY = "r0_spatial_mlx_export"
TRAIN_GAMES = (157, 156, 156, 156)
TRAIN_FIRST_GAME_INDEX = (200_000, 200_157, 200_313, 200_469)
VALIDATION_GAMES = (32, 31, 31, 31)
VALIDATION_FIRST_GAME_INDEX = (210_000, 210_032, 210_063, 210_094)
REQUIRED_SOURCE_FILES = {
    "CASCADIA_V2_GOAL.txt",
    "Cargo.lock",
    "Cargo.toml",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "tools/cluster_research_queue.py",
    "tools/r0_spatial_mlx_campaign.py",
    "tools/r0_spatial_mlx_report.py",
    "tools/rust_experiment_bundle.py",
}
REQUIRED_SOURCE_PREFIXES = (
    "apps/web/src/",
    "crates/cascadia-api/",
    "crates/cascadia-cli-v2/",
    "crates/cascadia-data/",
    "crates/cascadia-differential/",
    "crates/cascadia-eval/",
    "crates/cascadia-game/",
    "crates/cascadia-model/",
    "crates/cascadia-provenance/",
    "crates/cascadia-search/",
    "crates/cascadia-sim/",
    "legacy/crates/cascadia-ai/",
    "legacy/crates/cascadia-core/",
    "python/cascadia_mlx/",
)


class CampaignError(RuntimeError):
    """Raised when the R0 MLX campaign cannot proceed without ambiguity."""


@dataclass(frozen=True)
class DatasetPart:
    split: str
    part_index: int
    games: int
    first_game_index: int
    root: Path

    @property
    def records(self) -> int:
        return self.games * 80


def dataset_parts(dataset_root: Path = DEFAULT_DATASET_ROOT) -> tuple[DatasetPart, ...]:
    parts: list[DatasetPart] = []
    for split, games, first_indexes in (
        ("train", TRAIN_GAMES, TRAIN_FIRST_GAME_INDEX),
        ("validation", VALIDATION_GAMES, VALIDATION_FIRST_GAME_INDEX),
    ):
        for part_index in range(len(HOSTS)):
            parts.append(
                DatasetPart(
                    split=split,
                    part_index=part_index,
                    games=games[part_index],
                    first_game_index=first_indexes[part_index],
                    root=Path(f"{dataset_root}-{split}-part-{part_index}"),
                )
            )
    return tuple(parts)


def freeze_corpus(
    roots: list[Path] | None = None,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    """Validate all bytes and return the canonical frozen 60,000-row lock."""
    parts = dataset_parts(dataset_root)
    selected_roots = list(roots) if roots is not None else [part.root for part in parts]
    if len(selected_roots) != len(parts):
        raise CampaignError("the R0 MLX corpus requires exactly eight dataset roots")

    corpus_hasher = blake3.blake3()
    corpus_hasher.update(CORPUS_DIGEST_PREFIX)
    identities: list[dict[str, Any]] = []
    source_digest: str | None = None
    for order, (part, root) in enumerate(zip(parts, selected_roots, strict=True)):
        root = root.resolve()
        manifest_path = root / "dataset.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise CampaignError(f"cannot read dataset manifest {manifest_path}: {error}") from error
        if not isinstance(manifest, dict):
            raise CampaignError(f"dataset manifest must be an object: {manifest_path}")
        _validate_dataset_part(part, root, manifest)
        try:
            Dataset(root, verify_checksums=True)
        except ValueError as error:
            raise CampaignError(f"dataset validation failed at {root}: {error}") from error

        observed_source = manifest["provenance"]["v2_source_blake3"]
        if source_digest is None:
            source_digest = observed_source
        elif observed_source != source_digest:
            raise CampaignError("R0 corpus parts do not share one V2 source digest")

        corpus_hasher.update(len(manifest_bytes).to_bytes(8, "little"))
        corpus_hasher.update(manifest_bytes)
        identities.append(
            {
                "order": order,
                "split": part.split,
                "part_index": part.part_index,
                "root_name": root.name,
                "dataset_id": manifest["dataset_id"],
                "first_game_index": part.first_game_index,
                "completed_games": part.games,
                "total_records": part.records,
                "manifest_blake3": blake3.blake3(manifest_bytes).hexdigest(),
            }
        )

    identity = {
        "feature_schema": FEATURE_SCHEMA,
        "target_schema": TARGET_SCHEMA,
        "total_records": 60_000,
        "train_records": 50_000,
        "validation_records": 10_000,
        "source_v2_blake3": source_digest,
        "corpus_blake3": corpus_hasher.hexdigest(),
        "datasets": identities,
    }
    return {
        "schema_version": CORPUS_LOCK_SCHEMA_VERSION,
        "contract_id": CORPUS_LOCK_CONTRACT,
        "lock_id": canonical_blake3(identity),
        "identity": identity,
    }


def validate_corpus_lock(
    path: Path,
    *,
    roots: list[Path] | None = None,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    lock = _read_json(path, "corpus lock")
    if (
        lock.get("schema_version") != CORPUS_LOCK_SCHEMA_VERSION
        or lock.get("contract_id") != CORPUS_LOCK_CONTRACT
        or not isinstance(lock.get("identity"), dict)
        or canonical_blake3(lock["identity"]) != lock.get("lock_id")
    ):
        raise CampaignError("R0 MLX corpus lock is malformed or its identity drifted")
    identity = lock["identity"]
    if (
        identity.get("total_records") != 60_000
        or identity.get("train_records") != 50_000
        or identity.get("validation_records") != 10_000
        or len(identity.get("datasets", [])) != 8
    ):
        raise CampaignError("R0 MLX corpus lock does not name the frozen 60,000 rows")
    if roots is not None:
        observed = freeze_corpus(roots, dataset_root=dataset_root)
        if observed != lock:
            raise CampaignError("local corpus bytes do not match the supplied R0 lock")
    return lock


def validate_bundle_for_campaign(bundle: Path) -> dict[str, Any]:
    try:
        manifest = validate_bundle(bundle)
    except BundleError as error:
        raise CampaignError(str(error)) from error
    source_entries = manifest.get("identity", {}).get("source_files", [])
    source_paths = {
        entry.get("path")
        for entry in source_entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    missing_files = sorted(REQUIRED_SOURCE_FILES - source_paths)
    missing_prefixes = sorted(
        prefix
        for prefix in REQUIRED_SOURCE_PREFIXES
        if not any(path.startswith(prefix) for path in source_paths)
    )
    binaries = {
        entry.get("name"): entry
        for entry in manifest.get("identity", {}).get("binaries", [])
        if isinstance(entry, dict)
    }
    if missing_files or missing_prefixes or EXPORTER_BINARY not in binaries:
        raise CampaignError(
            "R0 MLX bundle is incomplete: "
            f"missing_files={missing_files}, missing_prefixes={missing_prefixes}, "
            f"missing_exporter={EXPORTER_BINARY not in binaries}"
        )
    return manifest


def create_authorization(
    *,
    bundle: Path,
    corpus_lock: Path,
    approved_by: str,
    approved_unix_ms: int | None = None,
) -> dict[str, Any]:
    if not approved_by.strip():
        raise CampaignError("authorization requires a nonempty approver")
    bundle = bundle.resolve()
    bundle_manifest = validate_bundle_for_campaign(bundle)
    lock = validate_corpus_lock(corpus_lock)
    source = source_provenance(bundle / "source")
    exporter = bundle / "bin" / EXPORTER_BINARY
    exporter_digest = file_blake3(exporter)
    binary_entry = next(
        entry
        for entry in bundle_manifest["identity"]["binaries"]
        if entry["name"] == EXPORTER_BINARY
    )
    if binary_entry["blake3"] != exporter_digest:
        raise CampaignError("exporter binary differs from the immutable bundle manifest")
    protocol = R0SpatialMlxTournamentProtocol().to_dict()
    identity = {
        "protocol_id": PROTOCOL_ID,
        "protocol_blake3": canonical_blake3(protocol),
        "corpus_lock_id": lock["lock_id"],
        "mlx_source_blake3": source["v2_source_blake3"],
        "exporter_executable_blake3": exporter_digest,
        "bundle_id": bundle_manifest["bundle_id"],
        "authorized_arms": list(AUTHORIZED_ARMS),
        "approved_by": approved_by.strip(),
        "approved_unix_ms": (
            time.time_ns() // 1_000_000 if approved_unix_ms is None else approved_unix_ms
        ),
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "approved": True,
        "authorization_id": canonical_blake3(identity),
        "identity": identity,
    }


def validate_authorization(
    path: Path,
    *,
    bundle: Path,
    corpus_lock: Path,
) -> dict[str, Any]:
    authorization = _read_json(path, "authorization")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("experiment_id") != EXPERIMENT_ID
        or authorization.get("adr") != ADR_ID
        or authorization.get("approved") is not True
        or not isinstance(authorization.get("identity"), dict)
        or canonical_blake3(authorization["identity"]) != authorization.get("authorization_id")
    ):
        raise CampaignError("R0 MLX production authorization is invalid")
    approved_unix_ms = authorization["identity"].get("approved_unix_ms")
    if (
        not isinstance(approved_unix_ms, int)
        or isinstance(approved_unix_ms, bool)
        or approved_unix_ms < 0
    ):
        raise CampaignError("R0 MLX authorization has an invalid approval timestamp")
    expected = create_authorization(
        bundle=bundle,
        corpus_lock=corpus_lock,
        approved_by=str(authorization["identity"].get("approved_by", "")),
        approved_unix_ms=approved_unix_ms,
    )
    if expected != authorization:
        raise CampaignError("R0 MLX authorization does not match the bundle or corpus")
    return authorization


def run_preflight(
    *,
    host: str,
    repository: Path,
    bundle: Path,
    corpus_lock: Path,
    authorization: Path,
    dataset_roots: list[Path],
) -> dict[str, Any]:
    if host not in HOSTS:
        raise CampaignError(f"unknown cluster host: {host}")
    repository = repository.resolve()
    bundle = bundle.resolve()
    if repository != bundle / "source":
        raise CampaignError("preflight repository must be the immutable bundle source")
    bundle_manifest = validate_bundle_for_campaign(bundle)
    lock = validate_corpus_lock(corpus_lock, roots=dataset_roots)
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
    )
    source = source_provenance(repository)

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    probe = mx.sum(mx.arange(1024, dtype=mx.float32))
    mx.eval(probe)
    device = str(mx.default_device())
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise CampaignError("R0 MLX production requires an Apple Silicon macOS host")
    if "gpu" not in device.lower():
        raise CampaignError("R0 MLX production requires the MLX GPU device")
    if source["v2_source_blake3"] != approval["identity"]["mlx_source_blake3"]:
        raise CampaignError("preflight source differs from the approved MLX source")

    scientific_identity = {
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "host": host,
        "bundle_id": bundle_manifest["bundle_id"],
        "authorization_id": approval["authorization_id"],
        "corpus_lock_id": lock["lock_id"],
        "source_v2_blake3": source["v2_source_blake3"],
        "exporter_executable_blake3": file_blake3(bundle / "bin" / EXPORTER_BINARY),
        "protocol_blake3": approval["identity"]["protocol_blake3"],
        "model_parameter_count": parameter_count(R0SpatialIsoValueModel()),
        "dataset_manifest_blake3": [
            entry["manifest_blake3"] for entry in lock["identity"]["datasets"]
        ],
        "mlx_version": version("mlx"),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "device": device,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "preflight_id": canonical_blake3(scientific_identity),
        "scientific_identity": scientific_identity,
        "operational": {
            "hostname": socket.gethostname().split(".")[0],
            "repository": str(repository),
            "bundle": str(bundle),
            "dataset_roots": [str(path.resolve()) for path in dataset_roots],
        },
        "checks": {
            "bundle_verified": True,
            "authorization_verified": True,
            "corpus_verified": True,
            "all_shards_checksummed": True,
            "apple_silicon_verified": True,
            "mlx_gpu_verified": True,
            "production_training_started": False,
        },
    }


def validate_preflight(
    path: Path,
    *,
    host: str,
    bundle: Path,
    corpus_lock: Path,
    authorization: Path,
) -> dict[str, Any]:
    report = _read_json(path, "preflight")
    identity = report.get("scientific_identity")
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
    )
    bundle_manifest = validate_bundle_for_campaign(bundle)
    lock = validate_corpus_lock(corpus_lock)
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("adr") != ADR_ID
        or not isinstance(identity, dict)
        or canonical_blake3(identity) != report.get("preflight_id")
        or identity.get("host") != host
        or identity.get("bundle_id") != bundle_manifest["bundle_id"]
        or identity.get("authorization_id") != approval["authorization_id"]
        or identity.get("corpus_lock_id") != lock["lock_id"]
        or identity.get("protocol_blake3") != approval["identity"]["protocol_blake3"]
        or identity.get("exporter_executable_blake3")
        != approval["identity"]["exporter_executable_blake3"]
        or not all(
            report.get("checks", {}).get(field) is True
            for field in (
                "bundle_verified",
                "authorization_verified",
                "corpus_verified",
                "all_shards_checksummed",
                "apple_silicon_verified",
                "mlx_gpu_verified",
            )
        )
        or report.get("checks", {}).get("production_training_started") is not False
    ):
        raise CampaignError("R0 MLX host preflight is absent, stale, or incomplete")
    return report


def run_arm(
    *,
    host: str,
    repository: Path,
    bundle: Path,
    corpus_lock: Path,
    authorization: Path,
    preflight: Path,
    dataset_roots: list[Path],
    arm: str,
    cache_root: Path,
    run_dir: Path,
    output: Path,
) -> dict[str, Any]:
    if arm not in ARM_ORDER:
        raise CampaignError(f"unknown R0 MLX arm: {arm}")
    if repository.resolve() != bundle.resolve() / "source":
        raise CampaignError("arm repository must be the immutable bundle source")
    resolved_host = detect_host(repository) if host == "auto" else host
    validate_preflight(
        preflight,
        host=resolved_host,
        bundle=bundle,
        corpus_lock=corpus_lock,
        authorization=authorization,
    )
    if output.exists():
        existing = _read_json(output, "existing arm report")
        _validate_arm_report(
            existing,
            arm=arm,
            authorization=validate_authorization(
                authorization,
                bundle=bundle,
                corpus_lock=corpus_lock,
            ),
        )
        return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = run_dir / "cache-export.json"
    command = [
        str(bundle.resolve() / "bin" / EXPORTER_BINARY),
        "--corpus-lock",
        str(corpus_lock.resolve()),
        *[value for root in dataset_roots for value in ("--dataset-root", str(root.resolve()))],
        "--arm",
        arm,
        "--output-root",
        str(cache_root.resolve()),
        "--receipt",
        str(receipt.resolve()),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise CampaignError(
            f"R0 MLX cache export failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    export = _read_json(receipt, "cache export receipt")
    if export.get("arm") != arm or export.get("experiment_id") != EXPERIMENT_ID:
        raise CampaignError("R0 MLX cache export receipt names the wrong arm")
    cache_path = Path(str(export["cache_root"]))
    R0SpatialMlxCache(cache_path, corpus_lock=corpus_lock)
    return run_tournament(
        R0SpatialMlxTournamentConfig(
            cache=cache_path,
            corpus_lock=corpus_lock,
            run_dir=run_dir / "checkpoints",
            output=output,
            authorization=authorization,
            resume=(run_dir / "checkpoints/latest.json").exists(),
        )
    )


def build_task_specs(
    *,
    repository: Path,
    bundle: Path,
    corpus_lock: Path,
    authorization: Path,
    queue: Path = DEFAULT_QUEUE,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> list[dict[str, Any]]:
    repository = repository.resolve()
    bundle = bundle.resolve()
    validate_bundle_for_campaign(bundle)
    validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
    )
    bundle_relative = _relative_to_repository(repository, bundle, "bundle")
    lock_relative = _relative_to_repository(repository, corpus_lock, "corpus lock")
    authorization_relative = _relative_to_repository(
        repository,
        authorization,
        "authorization",
    )
    experiment_relative = _relative_to_repository(
        repository,
        experiment_root,
        "experiment root",
    )
    queue_relative = _relative_to_repository(repository, queue, "queue")
    if corpus_lock.resolve().parent != authorization.resolve().parent:
        raise CampaignError("corpus lock and authorization must share one control directory")
    control_relative = lock_relative.parent

    bundle_fanout_id = f"{TASK_PREFIX}-bundle-fanout"
    control_fanout_id = f"{TASK_PREFIX}-control-fanout"
    specs = [
        _task(
            task_id=bundle_fanout_id,
            title="Fan out immutable R0 MLX bundle",
            decision="Install one byte-identical source and exporter bundle on all four hosts",
            workload_class="shared-prerequisite",
            priority=1,
            expected_runtime_seconds=180,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[],
            command=[
                ".venv/bin/python",
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{bundle_relative}/",
                "--local-root",
                str(bundle_relative),
                *[
                    value
                    for host in HOSTS[1:]
                    for value in (
                        "--destination",
                        f"{host}:{REMOTE_ROOTS[host] / bundle_relative}/",
                    )
                ],
                "--required-file",
                "bundle.json",
                "--verify-tree",
                "--output",
                str(experiment_relative / "reports/bundle-fanout.json"),
            ],
            artifact_path=str(experiment_relative / "reports/bundle-fanout.json"),
            stop_rule="Every source and binary byte must match the immutable bundle manifest.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        ),
        _task(
            task_id=control_fanout_id,
            title="Fan out R0 MLX approval controls",
            decision="Install the exact corpus lock and parent authorization on every host",
            workload_class="shared-prerequisite",
            priority=2,
            expected_runtime_seconds=30,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=[bundle_fanout_id],
            command=[
                ".venv/bin/python",
                "tools/cluster_artifact_fanout.py",
                "--source",
                f"{control_relative}/",
                "--local-root",
                str(control_relative),
                *[
                    value
                    for host in HOSTS[1:]
                    for value in (
                        "--destination",
                        f"{host}:{REMOTE_ROOTS[host] / control_relative}/",
                    )
                ],
                "--required-file",
                lock_relative.name,
                "--required-file",
                authorization_relative.name,
                "--verify-tree",
                "--output",
                str(experiment_relative / "reports/control-fanout.json"),
            ],
            artifact_path=str(experiment_relative / "reports/control-fanout.json"),
            stop_rule="All hosts must receive byte-identical lock and authorization files.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        ),
    ]

    preflight_ids: list[str] = []
    for host in HOSTS:
        preflight_id = f"{TASK_PREFIX}-preflight-{host}"
        preflight_ids.append(preflight_id)
        local_preflight = experiment_relative / "reports/preflight-local.json"
        specs.append(
            _task(
                task_id=preflight_id,
                title=f"Preflight R0 MLX on {host}",
                decision="Prove corpus, source, exporter, Apple Silicon, and MLX GPU readiness",
                workload_class="shared-prerequisite",
                priority=5,
                expected_runtime_seconds=180,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=[host],
                dependencies=[bundle_fanout_id, control_fanout_id],
                command=[
                    *_frozen_python_command(host, bundle_relative),
                    "preflight",
                    "--host",
                    host,
                    "--repository",
                    str(REMOTE_ROOTS[host] / bundle_relative / "source"),
                    "--bundle",
                    str(REMOTE_ROOTS[host] / bundle_relative),
                    "--corpus-lock",
                    str(REMOTE_ROOTS[host] / lock_relative),
                    "--authorization",
                    str(REMOTE_ROOTS[host] / authorization_relative),
                    *_dataset_flags(host, dataset_root),
                    "--output",
                    str(REMOTE_ROOTS[host] / local_preflight),
                ],
                artifact_path=str(local_preflight),
                stop_rule="No optimizer step may start unless every preflight check is true.",
                cpu_cores=1,
                memory_gib=2.0,
                uses_mlx=True,
            )
        )

    arm_task_ids: list[str] = []
    for arm, host in PRIMARY_ARMS:
        task_id = f"{TASK_PREFIX}-arm-{_slug(arm)}"
        arm_task_ids.append(task_id)
        specs.append(
            _arm_task(
                task_id=task_id,
                arm=arm,
                compatible_hosts=[host],
                dependencies=[f"{TASK_PREFIX}-preflight-{host}"],
                priority=10,
                bundle_relative=bundle_relative,
                lock_relative=lock_relative,
                authorization_relative=authorization_relative,
                experiment_relative=experiment_relative,
                dataset_root=dataset_root,
            )
        )
    historical_id = f"{TASK_PREFIX}-arm-{_slug(DIAGNOSTIC_ARM)}"
    arm_task_ids.append(historical_id)
    specs.append(
        _arm_task(
            task_id=historical_id,
            arm=DIAGNOSTIC_ARM,
            compatible_hosts=list(HOSTS),
            dependencies=preflight_ids,
            priority=20,
            bundle_relative=bundle_relative,
            lock_relative=lock_relative,
            authorization_relative=authorization_relative,
            experiment_relative=experiment_relative,
            dataset_root=dataset_root,
        )
    )

    collection_id = f"{TASK_PREFIX}-collect-reports"
    collection = experiment_relative / "reports/collection.json"
    specs.append(
        _task(
            task_id=collection_id,
            title="Collect all R0 MLX arm reports",
            decision="Retrieve each report from the host recorded by the work-conserving queue",
            workload_class="shared-prerequisite",
            priority=30,
            expected_runtime_seconds=120,
            critical_path=True,
            decision_terminal=False,
            compatible_hosts=["john1"],
            dependencies=arm_task_ids,
            command=[
                *_frozen_python_command("john1", bundle_relative),
                "collect-reports",
                "--queue",
                str(REMOTE_ROOTS["john1"] / queue_relative),
                "--authorization",
                str(REMOTE_ROOTS["john1"] / authorization_relative),
                "--bundle",
                str(REMOTE_ROOTS["john1"] / bundle_relative),
                "--corpus-lock",
                str(REMOTE_ROOTS["john1"] / lock_relative),
                "--output",
                str(REMOTE_ROOTS["john1"] / collection),
                "--destination",
                str(REMOTE_ROOTS["john1"] / experiment_relative / "reports/collected"),
            ],
            artifact_path=str(collection),
            stop_rule="Exactly five authorized arm reports must be retrieved and verified.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )

    classifier_ids = []
    for order in ("forward", "reverse"):
        task_id = f"{TASK_PREFIX}-classify-{order}"
        classifier_ids.append(task_id)
        output = experiment_relative / f"reports/classification-{order}.json"
        specs.append(
            _task(
                task_id=task_id,
                title=f"Classify R0 MLX reports in {order} order",
                decision="Apply ADR 0142 identity, integrity, quality, and throughput gates",
                workload_class="replica",
                priority=40,
                expected_runtime_seconds=30,
                critical_path=True,
                decision_terminal=False,
                compatible_hosts=["john1"],
                dependencies=[collection_id],
                command=[
                    *_frozen_report_command("john1", bundle_relative),
                    "--collection",
                    str(REMOTE_ROOTS["john1"] / collection),
                    "--order",
                    order,
                    "--output",
                    str(REMOTE_ROOTS["john1"] / output),
                ],
                artifact_path=str(output),
                stop_rule="Classification must be deterministic and must not authorize promotion.",
                cpu_cores=1,
                memory_gib=1.0,
                uses_mlx=False,
            )
        )

    comparison = experiment_relative / "reports/classification-order-proof.json"
    specs.append(
        _task(
            task_id=f"{TASK_PREFIX}-classification-order-proof",
            title="Prove R0 MLX classification order invariance",
            decision="Require byte-identical forward and reverse classifier outputs",
            workload_class="replica",
            priority=50,
            expected_runtime_seconds=10,
            critical_path=True,
            decision_terminal=True,
            compatible_hosts=["john1"],
            dependencies=classifier_ids,
            command=[
                *_frozen_python_command("john1", bundle_relative),
                "compare-classifications",
                "--forward",
                str(
                    REMOTE_ROOTS["john1"]
                    / experiment_relative
                    / "reports/classification-forward.json"
                ),
                "--reverse",
                str(
                    REMOTE_ROOTS["john1"]
                    / experiment_relative
                    / "reports/classification-reverse.json"
                ),
                "--output",
                str(REMOTE_ROOTS["john1"] / comparison),
            ],
            artifact_path=str(comparison),
            stop_rule="Forward and reverse aggregate bytes must match exactly.",
            cpu_cores=1,
            memory_gib=1.0,
            uses_mlx=False,
        )
    )
    _validate_task_specs(specs)
    return specs


def install_task_specs(queue: Path, specs: list[dict[str, Any]]) -> None:
    with locked_queue(queue) as state:
        existing = {task["id"] for task in state["tasks"]}
        duplicates = sorted(existing & {spec["id"] for spec in specs})
        if duplicates:
            raise CampaignError(f"queue already contains R0 MLX task IDs: {duplicates}")
        for spec in specs:
            add_task(state, spec)


def collect_reports(
    *,
    queue: Path,
    authorization: Path,
    bundle: Path,
    corpus_lock: Path,
    destination: Path,
) -> dict[str, Any]:
    state = load_queue(queue)
    approval = validate_authorization(
        authorization,
        bundle=bundle,
        corpus_lock=corpus_lock,
    )
    destination.mkdir(parents=True, exist_ok=True)
    collected = []
    for arm in ARM_ORDER:
        task_id = f"{TASK_PREFIX}-arm-{_slug(arm)}"
        task = next((value for value in state["tasks"] if value["id"] == task_id), None)
        if task is None or task.get("status") != "completed" or not task.get("result"):
            raise CampaignError(f"queue task is not complete: {task_id}")
        host = task["result"]["host"]
        artifact = task["result"].get("artifact")
        if host not in HOSTS or artifact != task["artifact_path"]:
            raise CampaignError(f"queue result provenance drifted for {task_id}")
        target = destination / f"{_slug(arm)}.json"
        _retrieve_artifact(host, Path(artifact), target)
        report = _read_json(target, f"{arm} report")
        _validate_arm_report(report, arm=arm, authorization=approval)
        collected.append(
            {
                "arm": arm,
                "task_id": task_id,
                "host": host,
                "queue_artifact": artifact,
                "file": str(target.resolve()),
                "blake3": file_blake3(target),
                "report_id": report["report_id"],
            }
        )
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "authorization_id": approval["authorization_id"],
        "reports": [
            {
                "arm": entry["arm"],
                "blake3": entry["blake3"],
                "report_id": entry["report_id"],
            }
            for entry in collected
        ],
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "collection_id": canonical_blake3(identity),
        "scientific_identity": identity,
        "reports": collected,
        "claims": {
            "all_reports_collected": True,
            "promotion_authorized": False,
        },
    }


def compare_classifications(forward: Path, reverse: Path) -> dict[str, Any]:
    try:
        forward_bytes = forward.read_bytes()
        reverse_bytes = reverse.read_bytes()
    except OSError as error:
        raise CampaignError(f"cannot read classifier output: {error}") from error
    if forward_bytes != reverse_bytes:
        raise CampaignError("forward and reverse R0 MLX classifications differ")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "adr": ADR_ID,
        "classification_blake3": blake3.blake3(forward_bytes).hexdigest(),
        "byte_identical": True,
        "promotion_authorized": False,
    }


def detect_host(repository: Path) -> str:
    resolved = repository.resolve()
    matches = []
    for host, root in REMOTE_ROOTS.items():
        try:
            resolved.relative_to(root)
            matches.append(host)
        except ValueError:
            pass
    if len(matches) != 1:
        raise CampaignError(f"cannot map repository path to one cluster host: {resolved}")
    return matches[0]


def canonical_blake3(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return blake3.blake3(payload).hexdigest()


def _validate_dataset_part(part: DatasetPart, root: Path, manifest: dict[str, Any]) -> None:
    expected_cards = {wildlife: "A" for wildlife in ("bear", "elk", "salmon", "hawk", "fox")}
    game = manifest.get("game")
    provenance = manifest.get("provenance")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("feature_schema") != FEATURE_SCHEMA
        or manifest.get("target_schema") != TARGET_SCHEMA
        or manifest.get("record_size") != 864
        or manifest.get("split") != part.split
        or manifest.get("strategy") != STRATEGY_ID
        or manifest.get("first_game_index") != part.first_game_index
        or manifest.get("requested_games") != part.games
        or manifest.get("completed_games") != part.games
        or manifest.get("total_records") != part.records
        or not isinstance(game, dict)
        or game.get("player_count") != 4
        or game.get("mode") != "Standard"
        or game.get("scoring_cards") != expected_cards
        or game.get("habitat_bonuses") is not False
        or not isinstance(provenance, dict)
    ):
        raise CampaignError(f"dataset part violates the frozen R0 contract: {root}")
    _require_digest(provenance.get("v2_source_blake3"), "dataset source digest")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise CampaignError(f"dataset part has no shard manifest: {root}")
    next_game = part.first_game_index
    total_games = 0
    total_records = 0
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("first_game_index") != next_game:
            raise CampaignError(f"dataset shard interval drifted: {root}")
        game_count = shard.get("game_count")
        record_count = shard.get("record_count")
        if (
            not isinstance(game_count, int)
            or game_count <= 0
            or not isinstance(record_count, int)
            or record_count != game_count * 80
        ):
            raise CampaignError(f"dataset shard row accounting drifted: {root}")
        next_game += game_count
        total_games += game_count
        total_records += record_count
    if total_games != part.games or total_records != part.records:
        raise CampaignError(f"dataset shard totals drifted: {root}")


def _validate_arm_report(
    report: dict[str, Any],
    *,
    arm: str,
    authorization: dict[str, Any],
) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("adr") != ADR_ID
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("arm") != arm
        or not isinstance(report.get("scientific_identity"), dict)
        or canonical_blake3(report["scientific_identity"]) != report.get("report_id")
        or report.get("authorization", {}).get("authorization_id")
        != authorization["authorization_id"]
        or report.get("integrity", {}).get("all_metrics_finite") is not True
        or report.get("claims", {}).get("promotion_authorized") is not False
    ):
        raise CampaignError(f"collected R0 MLX report is invalid: {arm}")


def _arm_task(
    *,
    task_id: str,
    arm: str,
    compatible_hosts: list[str],
    dependencies: list[str],
    priority: int,
    bundle_relative: Path,
    lock_relative: Path,
    authorization_relative: Path,
    experiment_relative: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    command_host = compatible_hosts[0]
    remote_root = REMOTE_ROOTS[command_host]
    report = experiment_relative / "runs" / arm / "report.json"
    command = [
        *_frozen_python_command(command_host, bundle_relative),
        "run-arm",
        "--host",
        "auto",
        "--repository",
        str(remote_root / bundle_relative / "source"),
        "--bundle",
        str(remote_root / bundle_relative),
        "--corpus-lock",
        str(remote_root / lock_relative),
        "--authorization",
        str(remote_root / authorization_relative),
        "--preflight",
        str(remote_root / experiment_relative / "reports/preflight-local.json"),
        *_dataset_flags(command_host, dataset_root),
        "--arm",
        arm,
        "--cache-root",
        str(remote_root / experiment_relative / "caches"),
        "--run-dir",
        str(remote_root / experiment_relative / "runs" / arm),
        "--output",
        str(remote_root / report),
    ]
    if len(compatible_hosts) > 1:
        command = [
            value.replace(str(remote_root), "__R0_HOST_ROOT__")
            if str(remote_root) in value
            else value
            for value in command
        ]
        command = [
            "/bin/zsh",
            "-lc",
            _portable_historical_shell(command),
        ]
    return _task(
        task_id=task_id,
        title=f"Train R0 MLX arm {arm}",
        decision="Measure one frozen iso-architecture representation arm",
        workload_class="independent-experiment",
        priority=priority,
        expected_runtime_seconds=3600,
        critical_path=True,
        decision_terminal=False,
        compatible_hosts=compatible_hosts,
        dependencies=dependencies,
        command=command,
        artifact_path=str(report),
        stop_rule="Complete 500 frozen optimizer steps and all validation/performance gates.",
        cpu_cores=8,
        memory_gib=12.0,
        uses_mlx=True,
    )


def _portable_historical_shell(command: list[str]) -> str:
    import shlex

    roots = " ".join(f"{host}:{root}" for host, root in REMOTE_ROOTS.items())
    rendered = []
    for value in command:
        if "__R0_HOST_ROOT__" not in value:
            rendered.append(shlex.quote(value))
            continue
        prefix, suffix = value.split("__R0_HOST_ROOT__", 1)
        rendered.append(f'{shlex.quote(prefix)}"$R0_ROOT"{shlex.quote(suffix)}')
    template = " ".join(rendered)
    return (
        'case "$PWD" in '
        + " ".join(f"{root}|{root}/*) R0_ROOT={root};;" for root in REMOTE_ROOTS.values())
        + f' *) echo "cannot map R0 host root; known={roots}" >&2; exit 2;; esac; '
        + template.replace("__R0_HOST_ROOT__", '"$R0_ROOT"')
    )


def _frozen_python_command(host: str, bundle_relative: Path) -> list[str]:
    root = REMOTE_ROOTS[host]
    return [
        "/usr/bin/env",
        "-C",
        str(root / bundle_relative / "source"),
        "PYTHONPATH=python",
        str(root / ".venv/bin/python"),
        "-B",
        "tools/r0_spatial_mlx_campaign.py",
    ]


def _frozen_report_command(host: str, bundle_relative: Path) -> list[str]:
    command = _frozen_python_command(host, bundle_relative)
    command[-1] = "tools/r0_spatial_mlx_report.py"
    return command


def _dataset_flags(host: str, dataset_root: Path) -> list[str]:
    return [
        value
        for part in dataset_parts(dataset_root)
        for value in ("--dataset-root", str(REMOTE_ROOTS[host] / part.root))
    ]


def _task(
    *,
    task_id: str,
    title: str,
    decision: str,
    workload_class: str,
    priority: int,
    expected_runtime_seconds: float,
    critical_path: bool,
    decision_terminal: bool,
    compatible_hosts: list[str],
    dependencies: list[str],
    command: list[str],
    artifact_path: str,
    stop_rule: str,
    cpu_cores: int,
    memory_gib: float,
    uses_mlx: bool,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "title": title,
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "workload_class": workload_class,
        "priority": priority,
        "decision_value": 1.0,
        "expected_runtime_seconds": expected_runtime_seconds,
        "critical_path": critical_path,
        "decision_terminal": decision_terminal,
        "compatible_hosts": compatible_hosts,
        "dependencies": dependencies,
        "command": command,
        "artifact_path": artifact_path,
        "stop_rule": stop_rule,
        "resources": {
            "cpu_cores": cpu_cores,
            "memory_gib": memory_gib,
            "uses_mlx": uses_mlx,
        },
    }


def _validate_task_specs(specs: list[dict[str, Any]]) -> None:
    identifiers = [spec["id"] for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise CampaignError("R0 MLX queue graph contains duplicate task IDs")
    known = set(identifiers)
    for spec in specs:
        unknown = set(spec["dependencies"]) - known
        if unknown:
            raise CampaignError(f"task {spec['id']} has unknown dependencies: {sorted(unknown)}")
    historical = next(
        spec for spec in specs if spec["id"] == f"{TASK_PREFIX}-arm-{_slug(DIAGNOSTIC_ARM)}"
    )
    if historical["compatible_hosts"] != list(HOSTS) or historical["priority"] <= 10:
        raise CampaignError("historical diagnostic does not preserve work-conserving backfill")


def _relative_to_repository(repository: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository)
    except ValueError as error:
        raise CampaignError(f"{label} must remain beneath the repository") from error
    if not relative.parts:
        raise CampaignError(f"{label} cannot be the repository root")
    return relative


def _retrieve_artifact(host: str, relative: Path, destination: Path) -> None:
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignError("queue artifact path must be repository-relative")
    source = REMOTE_ROOTS[host] / relative
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if host == "john1":
        try:
            shutil.copyfile(source, temporary)
        except OSError as error:
            raise CampaignError(f"cannot collect local report {source}: {error}") from error
    else:
        completed = subprocess.run(
            ["scp", f"{host}:{source}", str(temporary)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise CampaignError(
                f"cannot collect report from {host}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
    os.replace(temporary, destination)


def _slug(value: str) -> str:
    return (
        value.replace("historical-square-21x21-441", "historical441")
        .replace("exact-entity-control", "exact")
        .replace("hex-radius-", "r")
        .replace("-127", "")
        .replace("-91", "")
        .replace("-61", "")
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _write_once(path: Path, value: object, label: str) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    if path.exists():
        try:
            existing = path.read_text()
        except OSError as error:
            raise CampaignError(f"cannot read existing {label}: {error}") from error
        if existing != encoded:
            raise CampaignError(f"refusing to overwrite a different existing {label}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, value)


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignError(f"{label} must be a lowercase 64-character digest")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-corpus")
    freeze.add_argument("--dataset-root", type=Path, action="append")
    freeze.add_argument("--dataset-prefix", type=Path, default=DEFAULT_DATASET_ROOT)
    freeze.add_argument("--output", type=Path, default=DEFAULT_CORPUS_LOCK)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--bundle", type=Path, required=True)
    authorize.add_argument("--corpus-lock", type=Path, default=DEFAULT_CORPUS_LOCK)
    authorize.add_argument("--approved-by", required=True)
    authorize.add_argument("--output", type=Path, default=DEFAULT_AUTHORIZATION)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--host", required=True)
    preflight.add_argument("--repository", type=Path, required=True)
    preflight.add_argument("--bundle", type=Path, required=True)
    preflight.add_argument("--corpus-lock", type=Path, required=True)
    preflight.add_argument("--authorization", type=Path, required=True)
    preflight.add_argument("--dataset-root", type=Path, action="append", required=True)
    preflight.add_argument("--output", type=Path, required=True)

    arm = subparsers.add_parser("run-arm")
    arm.add_argument("--host", required=True)
    arm.add_argument("--repository", type=Path, required=True)
    arm.add_argument("--bundle", type=Path, required=True)
    arm.add_argument("--corpus-lock", type=Path, required=True)
    arm.add_argument("--authorization", type=Path, required=True)
    arm.add_argument("--preflight", type=Path, required=True)
    arm.add_argument("--dataset-root", type=Path, action="append", required=True)
    arm.add_argument("--arm", choices=ARM_ORDER, required=True)
    arm.add_argument("--cache-root", type=Path, required=True)
    arm.add_argument("--run-dir", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)

    queue = subparsers.add_parser("queue-spec")
    queue.add_argument("--repository", type=Path, default=Path("."))
    queue.add_argument("--bundle", type=Path, required=True)
    queue.add_argument("--corpus-lock", type=Path, default=DEFAULT_CORPUS_LOCK)
    queue.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    queue.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    queue.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    queue.add_argument("--dataset-prefix", type=Path, default=DEFAULT_DATASET_ROOT)
    queue.add_argument("--output", type=Path, required=True)
    queue.add_argument("--apply", action="store_true")

    collect = subparsers.add_parser("collect-reports")
    collect.add_argument("--queue", type=Path, required=True)
    collect.add_argument("--authorization", type=Path, required=True)
    collect.add_argument("--bundle", type=Path, required=True)
    collect.add_argument("--corpus-lock", type=Path, required=True)
    collect.add_argument("--destination", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare-classifications")
    compare.add_argument("--forward", type=Path, required=True)
    compare.add_argument("--reverse", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "freeze-corpus":
            report = freeze_corpus(args.dataset_root, dataset_root=args.dataset_prefix)
            _write_once(args.output, report, "corpus lock")
        elif args.command == "authorize":
            report = create_authorization(
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                approved_by=args.approved_by,
            )
            _write_once(args.output, report, "authorization")
        elif args.command == "preflight":
            report = run_preflight(
                host=args.host,
                repository=args.repository,
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                authorization=args.authorization,
                dataset_roots=args.dataset_root,
            )
            _write_json_atomic(args.output, report)
        elif args.command == "run-arm":
            report = run_arm(
                host=args.host,
                repository=args.repository,
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                authorization=args.authorization,
                preflight=args.preflight,
                dataset_roots=args.dataset_root,
                arm=args.arm,
                cache_root=args.cache_root,
                run_dir=args.run_dir,
                output=args.output,
            )
        elif args.command == "queue-spec":
            specs = build_task_specs(
                repository=args.repository,
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                authorization=args.authorization,
                queue=args.queue,
                experiment_root=args.experiment_root,
                dataset_root=args.dataset_prefix,
            )
            report = {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "adr": ADR_ID,
                "task_count": len(specs),
                "task_spec_blake3": canonical_blake3(specs),
                "applied": args.apply,
                "tasks": specs,
            }
            _write_json_atomic(args.output, report)
            if args.apply:
                install_task_specs(args.queue, specs)
        elif args.command == "collect-reports":
            report = collect_reports(
                queue=args.queue,
                authorization=args.authorization,
                bundle=args.bundle,
                corpus_lock=args.corpus_lock,
                destination=args.destination,
            )
            _write_json_atomic(args.output, report)
        else:
            report = compare_classifications(args.forward, args.reverse)
            _write_json_atomic(args.output, report)
    except CampaignError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
